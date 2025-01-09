#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import numpy as np

class LaneChangeNode:
    def __init__(self):
        rospy.init_node('lane_change_node', anonymous=False)
        
        # 基本参数设置
        self.bridge = CvBridge()
        self.lane_change_threshold = 1.0  # 触发变道的距离阈值(米)
        self.lane_width = 0.6  # 变道距离(米)
        
        # 图像处理参数
        self.known_width = 0.3  # 假设障碍物宽度(米)
        self.focal_length = 800  # 相机焦距(像素)，需要标定
        
        # 参数初始化
        self.obstacle_distance = float('inf')
        self.current_x = 0.0  # 当前位置
        self.current_y = 0.0
        self.start_change_x = 0.0  # 开始变道时的位置
        self.start_change_y = 0.0
        
        # 状态机
        self.STATE_FORWARD = 0
        self.STATE_CHANGING = 1
        self.current_state = self.STATE_FORWARD
        
        # 运动参数
        self.linear_speed = 0.3
        self.angular_speed = 0.3
        
        # 修改订阅器
        self.image_sub = rospy.Subscriber("/camera/image_raw", Image, self.image_callback)
        self.odom_sub = rospy.Subscriber("/odom", Odometry, self.odom_callback)
        self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        
        # 添加图像处理参数
        self.object_detector = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=40)
        
        rospy.loginfo("基于视觉的车道变换节点已初始化")

    def estimate_distance(self, pixel_width):
        """使用单目视觉估算距离"""
        if pixel_width > 0:
            return (self.known_width * self.focal_length) / pixel_width
        return float('inf')

    def detect_obstacle(self, frame):
        """检测前方障碍物"""
        height, width = frame.shape[:2]
        
        # 定义感兴趣区域 (ROI)
        roi_height = height // 2
        roi_y = height // 3
        roi = frame[roi_y:roi_y+roi_height, :]
        
        # 图像预处理
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 使用背景分割器检测运动物体
        mask = self.object_detector.apply(blur)
        _, mask = cv2.threshold(mask, 250, 255, cv2.THRESH_BINARY)
        
        # 形态学操作改善检测效果
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        min_distance = float('inf')
        obstacle_detected = False
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 400:  # 过滤小物体
                x, y, w, h = cv2.boundingRect(cnt)
                distance = self.estimate_distance(w)
                
                if distance < min_distance:
                    min_distance = distance
                    obstacle_detected = True
                
                # 在原图上绘制检测框和距离信息
                cv2.rectangle(frame, (x, y+roi_y), (x+w, y+h+roi_y), (0, 255, 0), 2)
                cv2.putText(frame, f"D: {distance:.2f}m", (x, y+roi_y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 显示处理后的图像
        cv2.imshow("Object Detection", frame)
        cv2.imshow("Mask", mask)
        cv2.waitKey(1)
        
        return min_distance, obstacle_detected

    def image_callback(self, msg):
        """处理图像回调"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            distance, detected = self.detect_obstacle(cv_image)
            
            if detected:
                self.obstacle_distance = distance
                self.check_and_control()
            
        except Exception as e:
            rospy.logerr("图像处理错误: %s", str(e))

    def odom_callback(self, data):
        """处理里程计数据"""
        self.current_x = data.pose.pose.position.x
        self.current_y = data.pose.pose.position.y

    def check_and_control(self):
        """检查状态并发送控制命令"""
        cmd = Twist()
        
        if self.current_state == self.STATE_FORWARD:
            if self.obstacle_distance < self.lane_change_threshold:
                # 检测到障碍物，开始变道
                self.start_change_x = self.current_x
                self.start_change_y = self.current_y
                self.current_state = self.STATE_CHANGING
                rospy.loginfo("检测到障碍物 %.2f米, 开始变道", self.obstacle_distance)
            else:
                # 继续直行，速度与障碍物距离相关
                cmd.linear.x = min(self.linear_speed, 
                                 self.linear_speed * (self.obstacle_distance / self.lane_change_threshold))
                cmd.angular.z = 0.0
        
        elif self.current_state == self.STATE_CHANGING:
            # 变道状态
            lateral_distance = abs(self.current_y - self.start_change_y)
            
            if lateral_distance < self.lane_width:
                # 正在变道
                cmd.linear.x = self.linear_speed * 0.8  # 变道时降低速度
                cmd.angular.z = self.angular_speed
                rospy.loginfo("变道中... 当前横向位移: %.2f 米", lateral_distance)
            else:
                # 变道完成，恢复直行
                self.current_state = self.STATE_FORWARD
                cmd.linear.x = self.linear_speed
                cmd.angular.z = 0.0
                rospy.loginfo("变道完成，恢复直行")
        
        # 发布控制命令
        self.cmd_vel_pub.publish(cmd)

    def run(self):
        """主循环"""
        rate = rospy.Rate(10)  # 10Hz
        while not rospy.is_shutdown():
            rate.sleep()

if __name__ == '__main__':
    try:
        node = LaneChangeNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
