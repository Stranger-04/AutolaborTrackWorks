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
            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            
            # 初始化mask变量
            mask = np.zeros(cv_image.shape[:2], dtype=np.uint8)
            
            if self.tracking_roi and self.selected_roi:
                x, y, w, h = self.selected_roi
                cv2.rectangle(cv_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                roi_hsv = hsv[y:y+h, x:x+w]
                if roi_hsv.size > 0:
                    # 获取ROI区域的HSV值范围
                    roi_hist = cv2.calcHist([roi_hsv], [0], None, [180], [0, 180])
                    roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
                    
                    # 反向投影
                    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
                    
                    # 使用均值漂移跟踪
                    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
                    ret, self.selected_roi = cv2.CamShift(dst, self.selected_roi, term_crit)
                    
                    # 更新中心点位置
                    cx = x + w//2
                    cy = y + h//2
                    
                    # 控制逻辑
                    error_x = cx - self.image_center_x
                    control_info = "Error_X: %.2f" % error_x
                    cv2.putText(cv_image, control_info, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # 创建运动控制消息
                    cmd = Twist()
                    angular_z = -self.angular_speed * (error_x / float(self.image_center_x))
                    cmd_info = "Angular: %.2f" % angular_z
                    cv2.putText(cv_image, cmd_info, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # 根据误差控制机器人运动
                    if abs(error_x) > self.center_threshold:
                        # 转向控制
                        cmd.angular.z = angular_z
                    else:
                        # 前进控制
                        cmd.linear.x = self.linear_speed
                        cmd_info = "Linear: %.2f" % self.linear_speed
                        cv2.putText(cv_image, cmd_info, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    rospy.loginfo("目标位置: (%d, %d), 面积: %.2f", cx, cy, area)
                    
                    # 更新mask为反向投影结果
                    mask = dst
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
