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
        
        # 修改控制参数以匹配vision_node
        self.bridge = CvBridge()
        self.target_color = rospy.get_param('~target_color', 'red')
        self.linear_speed = rospy.get_param('~linear_speed', 0.3)   # 增加默认速度
        self.angular_speed = rospy.get_param('~angular_speed', 0.5) # 增加角速度
        self.min_area = rospy.get_param('~min_area', 500)
        self.image_center_x = 320
        self.center_threshold = 100  # 增加阈值范围，使运动更平滑
        
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

        # 添加框选相关变量
        self.drawing = False
        self.roi_points = []
        self.selected_roi = None
        self.tracking_roi = False
        
        # 添加鼠标回调
        cv2.setMouseCallback('camera_view', self.mouse_callback)

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

    def mouse_callback(self, event, x, y, flags, param):
        """处理鼠标事件"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.roi_points = [(x, y)]
            self.tracking_roi = False
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            frame_copy = self.current_frame.copy()
            cv2.rectangle(frame_copy, self.roi_points[0], (x, y), (0, 255, 0), 2)
            cv2.imshow('camera_view', frame_copy)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.roi_points.append((x, y))
            self.selected_roi = self.get_roi_coordinates()
            self.tracking_roi = True

    def get_roi_coordinates(self):
        """获取ROI坐标"""
        if len(self.roi_points) == 2:
            x1, y1 = self.roi_points[0]
            x2, y2 = self.roi_points[1]
            return (min(x1, x2), min(y1, y2), abs(x2-x1), abs(y2-y1))
        return None

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.current_frame = cv_image.copy()
            
            # 图像预处理优化
            # 1. 高斯模糊去噪
            cv_image = cv2.GaussianBlur(cv_image, (5, 5), 0)
            
            # 2. 使用自适应直方图均衡化
            lab = cv2.cvtColor(cv_image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            lab = cv2.merge((l,a,b))
            cv_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            # 3. 增强对比度
            alpha = 1.3  # 对比度因子
            beta = 10    # 亮度增强
            cv_image = cv2.convertScaleAbs(cv_image, alpha=alpha, beta=beta)
            
            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            mask = np.zeros(cv_image.shape[:2], dtype=np.uint8)
            
            if self.tracking_roi and self.selected_roi:
                x, y, w, h = self.selected_roi
                roi_hsv = hsv[y:y+h, x:x+w]
                
                if roi_hsv.size > 0:
                    # 提取ROI的颜色特征
                    roi_mean = cv2.mean(roi_hsv)[:3]
                    h_mean, s_mean, v_mean = roi_mean
                    
                    # 创建动态HSV范围
                    h_range = 20
                    s_range = 50
                    v_range = 50
                    
                    lower_bound = np.array([max(0, h_mean - h_range), 
                                          max(0, s_mean - s_range),
                                          max(0, v_mean - v_range)])
                    upper_bound = np.array([min(180, h_mean + h_range),
                                          min(255, s_mean + s_range),
                                          min(255, v_mean + v_range)])
                    
                    # 创建目标掩码
                    target_mask = cv2.inRange(hsv, lower_bound, upper_bound)
                    
                    # 形态学操作
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
                    target_mask = cv2.morphologyEx(target_mask, cv2.MORPH_CLOSE, kernel)
                    target_mask = cv2.morphologyEx(target_mask, cv2.MORPH_OPEN, kernel)
                    
                    # 找到最大连通区域 - 兼容不同版本OpenCV
                    im2, contours, hierarchy = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        largest_contour = max(contours, key=cv2.contourArea)
                        area = cv2.contourArea(largest_contour)
                        if area > self.min_area:
                            x, y, w, h = cv2.boundingRect(largest_contour)
                            self.selected_roi = (x, y, w, h)
                            
                            # 绘制跟踪框和中心点
                            cv2.rectangle(cv_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            cx = x + w//2
                            cy = y + h//2
                            cv2.circle(cv_image, (cx, cy), 5, (0, 0, 255), -1)
                            
                            # 更新控制命令
                            error_x = cx - self.image_center_x
                            cmd = Twist()
                            
                            # 修改控制逻辑
                            if abs(error_x) > self.center_threshold:
                                # 目标不在中心区域，需要转向
                                cmd.angular.z = -self.angular_speed * (error_x / float(self.image_center_x))
                                cmd.linear.x = 0.0  # 转向时停止前进
                                rospy.loginfo("转向控制: angular.z = %.2f", cmd.angular.z)
                            else:
                                # 目标在中心区域，可以前进
                                cmd.linear.x = min(self.linear_speed * (1 - abs(error_x)/float(self.image_center_x)), 
                                                 self.linear_speed)  # 根据偏差调整速度
                                cmd.angular.z = 0.0
                                rospy.loginfo("前进控制: linear.x = %.2f", cmd.linear.x)
                            
                            # 发布控制命令
                            self.cmd_vel_pub.publish(cmd)
                            
                            # 显示控制信息
                            cv2.putText(cv_image, "Error_X: %.2f" % error_x, (10, 60), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            cv2.putText(cv_image, "Cmd: lin=%.2f, ang=%.2f" % (cmd.linear.x, cmd.angular.z),
                                      (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            
                            rospy.loginfo("目标位置: (%d, %d), 面积: %.2f", cx, cy, area)
                            
                            # 更新mask为目标掩码
                            mask = target_mask
                        else:
                            cv2.putText(cv_image, "Target Lost", (10, 180), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
            else:
                # 如果没有ROI追踪，使用颜色掩码
                mask = self.get_color_mask(hsv)
            
            # 显示提示信息
            cv2.putText(cv_image, "Press 'r' to reset tracking", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 显示图像和掩码
            cv2.imshow('camera_view', cv_image)
            cv2.imshow('mask', mask)
            key = cv2.waitKey(3) & 0xFF
            if key == ord('r'):
                self.tracking_roi = False
                self.selected_roi = None
            
        except ValueError as ve:
            # 处理OpenCV 4.x版本
            try:
                contours, hierarchy = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(largest_contour)
                    if area > self.min_area:
                        x, y, w, h = cv2.boundingRect(largest_contour)
                        self.selected_roi = (x, y, w, h)
                        
                        # 绘制跟踪框和中心点
                        cv2.rectangle(cv_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cx = x + w//2
                        cy = y + h//2
                        cv2.circle(cv_image, (cx, cy), 5, (0, 0, 255), -1)
                        
                        # 更新控制命令
                        error_x = cx - self.image_center_x
                        cmd = Twist()
                        
                        # 修改控制逻辑
                        if abs(error_x) > self.center_threshold:
                            # 目标不在中心区域，需要转向
                            cmd.angular.z = -self.angular_speed * (error_x / float(self.image_center_x))
                            cmd.linear.x = 0.0  # 转向时停止前进
                            rospy.loginfo("转向控制: angular.z = %.2f", cmd.angular.z)
                        else:
                            # 目标在中心区域，可以前进
                            cmd.linear.x = min(self.linear_speed * (1 - abs(error_x)/float(self.image_center_x)), 
                                             self.linear_speed)  # 根据偏差调整速度
                            cmd.angular.z = 0.0
                            rospy.loginfo("前进控制: linear.x = %.2f", cmd.linear.x)
                        
                        # 发布控制命令
                        self.cmd_vel_pub.publish(cmd)
                        
                        # 显示控制信息
                        cv2.putText(cv_image, "Error_X: %.2f" % error_x, (10, 60), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(cv_image, "Cmd: lin=%.2f, ang=%.2f" % (cmd.linear.x, cmd.angular.z),
                                  (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        rospy.loginfo("目标位置: (%d, %d), 面积: %.2f", cx, cy, area)
                        
                        # 更新mask为目标掩码
                        mask = target_mask
                    else:
                        cv2.putText(cv_image, "Target Lost", (10, 180), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            except Exception as e2:
                rospy.logerr("轮廓查找错误: %s", str(e2))
                
        except Exception as e:
            rospy.logerr("图像处理错误: %s", str(e))
            import traceback
            rospy.logerr(traceback.format_exc())  # 添加详细错误信息

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
