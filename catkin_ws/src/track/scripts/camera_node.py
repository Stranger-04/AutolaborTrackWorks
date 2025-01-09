#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import numpy as np

class HumanTracker:
    def __init__(self):
        rospy.init_node('human_tracker', anonymous=False)
        
        # 基本参数设置
        self.bridge = CvBridge()
        self.linear_speed = rospy.get_param('~linear_speed', 0.3)
        self.angular_speed = rospy.get_param('~angular_speed', 0.5)
        self.min_area = rospy.get_param('~min_area', 500)
        self.image_center_x = 320
        self.center_threshold = 100
        
        # 初始化HOG行人检测器
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        # 创建显示窗口
        cv2.namedWindow('camera_view', cv2.WINDOW_NORMAL)
        
        # ROS订阅和发布
        self.image_sub = rospy.Subscriber("/camera/image_raw", Image, self.image_callback)
        self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        
        # 跟踪状态
        self.tracking_target = None

    def detect_human(self, frame):
        """使用HOG检测器检测人形目标"""
        # 调整图像大小以提高检测速度
        scale = 0.5
        small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        
        # 检测人形
        boxes, weights = self.hog.detectMultiScale(
            small_frame,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05,
            hitThreshold=0
        )
        
        # 将检测结果缩放回原始尺寸
        boxes = np.array([[x/scale, y/scale, w/scale, h/scale] for (x, y, w, h) in boxes])
        
        return boxes, weights

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # 图像预处理
            cv_image = cv2.GaussianBlur(cv_image, (5, 5), 0)
            
            # 检测人形
            boxes, weights = self.detect_human(cv_image)
            
            if len(boxes) > 0:
                # 选择置信度最高的检测结果
                max_weight_idx = np.argmax(weights)
                x, y, w, h = boxes[max_weight_idx].astype(int)
                
                # 绘制检测框
                cv2.rectangle(cv_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # 计算目标中心
                cx = x + w//2
                cy = y + h//2
                cv2.circle(cv_image, (cx, cy), 5, (0, 0, 255), -1)
                
                # 更新控制命令
                error_x = cx - self.image_center_x
                cmd = Twist()
                
                if abs(error_x) > self.center_threshold:
                    # 目标不在中心区域，需要转向
                    cmd.angular.z = -self.angular_speed * (error_x / float(self.image_center_x))
                    cmd.linear.x = 0.0
                    rospy.loginfo("转向跟踪人形目标: angular.z = %.2f", cmd.angular.z)
                else:
                    # 目标在中心区域，前进
                    distance_factor = h / float(cv_image.shape[0])  # 使用目标高度估算距离
                    cmd.linear.x = self.linear_speed * min(1.0, distance_factor)
                    cmd.angular.z = 0.0
                    rospy.loginfo("接近人形目标: linear.x = %.2f", cmd.linear.x)
                
                # 发布控制命令
                self.cmd_vel_pub.publish(cmd)
                
                # 显示控制信息
                cv2.putText(cv_image, "Human detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(cv_image, "Cmd: lin=%.2f, ang=%.2f" % (cmd.linear.x, cmd.angular.z),
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                # 未检测到目标
                cv2.putText(cv_image, "No human detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # 停止移动
                cmd = Twist()
                self.cmd_vel_pub.publish(cmd)
            
            # 显示图像
            cv2.imshow('camera_view', cv_image)
            cv2.waitKey(3)
            
        except Exception as e:
            rospy.logerr("图像处理错误: %s", str(e))
            import traceback
            rospy.logerr(traceback.format_exc())

    def cleanup(self):
        """清理资源"""
        cv2.destroyAllWindows()
        rospy.loginfo("正在关闭人形跟踪节点...")

if __name__ == '__main__':
    try:
        tracker = HumanTracker()
        rospy.on_shutdown(tracker.cleanup)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
