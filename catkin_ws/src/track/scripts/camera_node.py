#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import numpy as np

class ColorTracker:
    def __init__(self):
        rospy.init_node('color_tracker', anonymous=False)
        
        # 参数设置
        self.bridge = CvBridge()
        self.target_color = rospy.get_param('~target_color', 'red')  # 默认跟踪红色
        self.linear_speed = rospy.get_param('~linear_speed', 0.2)
        self.angular_speed = rospy.get_param('~angular_speed', 0.2)
        self.min_area = rospy.get_param('~min_area', 500)  # 最小目标面积
        self.image_center_x = 320  # 图像中心x坐标
        self.center_threshold = 50  # 中心区域阈值
        
        # 颜色范围定义 (HSV空间)
        self.color_ranges = {
            'red': [np.array([0,100,100]), np.array([10,255,255])],  # 红色范围1
            'red2': [np.array([160,100,100]), np.array([179,255,255])],  # 红色范围2
            'blue': [np.array([100,100,100]), np.array([130,255,255])],
            'green': [np.array([50,100,100]), np.array([70,255,255])]
        }

        # 创建窗口
        cv2.namedWindow('camera_view', cv2.WINDOW_NORMAL)
        cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
        
        # ROS订阅和发布
        self.image_sub = rospy.Subscriber("/camera/image_raw", Image, self.image_callback)
        self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)

    def get_color_mask(self, hsv_image):
        """获取指定颜色的掩码"""
        if self.target_color == 'red':
            # 红色需要合并两个区间
            mask1 = cv2.inRange(hsv_image, self.color_ranges['red'][0], self.color_ranges['red'][1])
            mask2 = cv2.inRange(hsv_image, self.color_ranges['red2'][0], self.color_ranges['red2'][1])
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            # 其他颜色使用单个区间
            color_range = self.color_ranges.get(self.target_color)
            if color_range:
                mask = cv2.inRange(hsv_image, color_range[0], color_range[1])
            else:
                rospy.logwarn("不支持的颜色: %s", self.target_color)
                mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
        
        # 形态学操作改善掩码质量
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def image_callback(self, msg):
        try:
            # 转换ROS图像消息到OpenCV格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            
            # 获取颜色掩码
            mask = self.get_color_mask(hsv)
            
            # 添加调试信息显示
            debug_info = f"目标颜色: {self.target_color}"
            cv2.putText(cv_image, debug_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 修改findContours调用以兼容OpenCV 3.x
            _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 创建运动控制消息
            cmd = Twist()
            
            if contours:
                # 获取最大轮廓
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                
                if area > self.min_area:
                    # 计算轮廓的中心点
                    M = cv2.moments(largest_contour)
                    if M['m00'] > 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        
                        # 绘制轮廓和中心点
                        cv2.drawContours(cv_image, [largest_contour], -1, (0,255,0), 2)
                        cv2.circle(cv_image, (cx,cy), 5, (0,0,255), -1)
                        
                        # 计算偏离中心的误差
                        error_x = cx - self.image_center_x
                        
                        # 添加控制参数显示
                        control_info = f"Error_X: {error_x:.2f}"
                        cv2.putText(cv_image, control_info, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        cmd_info = f"Angular: {-self.angular_speed * (error_x / float(self.image_center_x)):.2f}"
                        cv2.putText(cv_image, cmd_info, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # 根据误差控制机器人运动
                        if abs(error_x) > self.center_threshold:
                            # 转向控制
                            cmd.angular.z = -self.angular_speed * (error_x / float(self.image_center_x))
                        else:
                            # 前进控制
                            cmd.linear.x = self.linear_speed
                            cmd_info = f"Linear: {self.linear_speed:.2f}"
                            cv2.putText(cv_image, cmd_info, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        rospy.loginfo("目标位置: (%d, %d), 面积: %.2f", cx, cy, area)
            
            # 发布控制命令
            self.cmd_vel_pub.publish(cmd)
            
            # 显示图像
            cv2.imshow('camera_view', cv_image)
            cv2.imshow('mask', mask)
            cv2.waitKey(3)
            
        except Exception as e:
            rospy.logerr("图像处理错误: %s", str(e))

    def cleanup(self):
        """清理资源"""
        cv2.destroyAllWindows()
        rospy.loginfo("正在关闭颜色跟踪节点...")

if __name__ == '__main__':
    try:
        tracker = ColorTracker()
        rospy.on_shutdown(tracker.cleanup)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
