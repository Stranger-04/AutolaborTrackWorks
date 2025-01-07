#!/usr/bin/env python
# -*- coding: utf-8 -*-
#############
# 视觉跟踪节点：通过OpenCV进行目标选择和方向计算
#############
import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import numpy as np  # 修复了numpy导入语句
import math
from geometry_msgs.msg import Twist
xs, ys, ws, hs = 0, 0, 0, 0  # selection.x selection.y
xo, yo = 0, 0  # origin.x origin.y
selectObject = False
trackObject = 0
##################
# 鼠标事件回调函数
# 功能：处理目标区域的选择
# event: 鼠标事件类型
# x, y: 鼠标坐标
# flags: 鼠标事件标志
##################
def onMouse(event, x, y, flags, prams):
    global xs, ys, ws, hs, selectObject, xo, yo, trackObject
    if event == cv2.EVENT_LBUTTONDOWN:
        xo, yo = x, y
        xs, ys, ws, hs = x, y, 0, 0
        selectObject = True
        rospy.loginfo("开始框选: (%d, %d)", x, y)  # 添加调试信息
    elif event == cv2.EVENT_MOUSEMOVE:
        if selectObject:  # 鼠标移动时更新框选区域
            xs = min(x, xo)
            ys = min(y, yo)
            ws = abs(x - xo)
            hs = abs(y - yo)
            rospy.loginfo("框选范围: x=%d, y=%d, w=%d, h=%d", xs, ys, ws, hs)  # 添加调试信息
    elif event == cv2.EVENT_LBUTTONUP:
        selectObject = False
        if ws > 0 and hs > 0:
            trackObject = -1
            rospy.loginfo("完成框选: w=%d, h=%d", ws, hs)  # 添加调试信息
        else:
            rospy.logwarn("框选区域太小，请重新选择")
##################
# CamShift目标跟踪实现
# 功能：
# 1. 将图像转换为HSV色彩空间
# 2. 根据选定区域创建直方图
# 3. 使用反向投影和CamShift算法跟踪目标
# 返回值：
# centerX: 目标中心x坐标
# length_of_diagonal: 目标框对角线长度
##################
def ExamByCamshift():
    global xs, ys, ws, hs, selectObject, xo, yo, trackObject, image, roi_hist, track_window
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    centerX = -1.0
    length_of_diagonal = float()

    try:
        # 显示当前图像尺寸和模式
        rospy.loginfo("图像尺寸: %dx%d", image.shape[1], image.shape[0])
        
        # 处理框选
        if selectObject and ws > 0 and hs > 0:
            # 显示选择框
            cv2.rectangle(image, (xs, ys), (xs+ws), (ys+hs), (0, 255, 0), 2)
            # 反色显示选择区域
            cv2.bitwise_not(image[ys:ys + hs, xs:xs + ws], image[ys:ys + hs, xs:xs + ws])
            rospy.loginfo("选择区域: x=%d, y=%d, w=%d, h=%d", xs, ys, ws, hs)

        # 跟踪处理
        if trackObject != 0:
            # 创建更宽松的掩码范围以适应Gazebo中的颜色
            mask = cv2.inRange(hsv, np.array((20., 100., 100.)), np.array((80., 255., 255.)))
            
            if trackObject == -1:  # 初始化追踪
                # 确保选择区域有效
                if ws > 0 and hs > 0:
                    track_window = (xs, ys, ws, hs)
                    roi_mask = mask[ys:ys + hs, xs:xs + ws]
                    roi_hsv = hsv[ys:ys + hs, xs:xs + ws]
                    
                    # 计算ROI区域的HSV直方图，使用3个通道
                    roi_hist = cv2.calcHist([roi_hsv], [0, 1, 2], roi_mask, [16, 16, 16], 
                                            [0, 180, 0, 256, 0, 256])
                    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
                    
                    # 输出ROI区域的HSV范围
                    rospy.loginfo("ROI HSV范围 - H: %d-%d, S: %d-%d, V: %d-%d",
                                np.min(roi_hsv[:,:,0]), np.max(roi_hsv[:,:,0]),
                                np.min(roi_hsv[:,:,1]), np.max(roi_hsv[:,:,1]),
                                np.min(roi_hsv[:,:,2]), np.max(roi_hsv[:,:,2]))
                    
                    trackObject = 1
                    
                    # 显示选择区域的HSV信息以帮助调试
                    mean_hsv = cv2.mean(roi_hsv, roi_mask)
                    rospy.loginfo("目标区域HSV平均值: H=%.1f, S=%.1f, V=%.1f", 
                                 mean_hsv[0], mean_hsv[1], mean_hsv[2])
            
            if trackObject == 1:  # 执行追踪
                # 计算反向投影
                dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
                dst &= mask
                
                # 应用CamShift
                ret, track_window = cv2.CamShift(dst, track_window, term_crit)
                
                # 检查跟踪结果是否有效
                if track_window[2] > 0 and track_window[3] > 0:
                    centerX = float(ret[0][0])
                    pts = cv2.boxPoints(ret)
                    pts = np.int0(pts)
                    length_of_diagonal = math.sqrt(float(ret[1][1]) ** 2 + float(ret[1][0]) ** 2)
                    
                    # 绘制跟踪结果
                    cv2.polylines(image, [pts], True, (0, 0, 255), 2)
                    cv2.circle(image, (int(centerX), int(ret[0][1])), 5, (0, 255, 255), -1)
                    
                    # 显示跟踪信息
                    rospy.loginfo("跟踪位置: (%.1f, %.1f), 大小: %.1f", centerX, ret[0][1], length_of_diagonal)
                else:
                    rospy.logwarn("跟踪窗口无效")
                    centerX = -1.0

        cv2.imshow('imshow', image)
        cv2.waitKey(3)
        
    except Exception as e:
        rospy.logerr("跟踪失败: %s", str(e))
        return -1.0, 0.0
    
    return centerX, length_of_diagonal

class image_listenner:
    def __init__(self):
        # 调整参数
        self.threshold = 80  # 增加阈值
        self.linear_x = 0.3
        self.angular_z = 0.2
        self.track_windows_threshold = math.sqrt(95*95+235*235)+10000
        
        # 初始化
        cv2.namedWindow('imshow', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('imshow', 800, 600)
        cv2.setMouseCallback('imshow', onMouse)
        self.bridge = CvBridge()
        
        # 更新相机话题名称
        rgb_topic = "/track_car/camera/image_raw"
        depth_topic = "/track_car/camera/depth/image_raw"
        
        rospy.loginfo("订阅RGB相机话题: %s", rgb_topic)
        rospy.loginfo("订阅深度相机话题: %s", depth_topic)
        
        # 订阅相机话题
        self.image_sub = rospy.Subscriber(rgb_topic, Image, self.image_sub_callback, queue_size=1)
        self.depth_sub = rospy.Subscriber(depth_topic, Image, self.depth_callback, queue_size=1)
        
        # 等待相机话题
        try:
            rospy.wait_for_message(rgb_topic, Image, timeout=5.0)
            rospy.wait_for_message(depth_topic, Image, timeout=5.0)
            rospy.loginfo("成功接收到相机数据")
        except rospy.ROSException:
            rospy.logerr("无法接收相机数据，请检查相机话题")
            raise

        self.twist_pub = rospy.Publisher("cmd_vel", Twist, queue_size=1)

        self.depth_image = None
        self.target_depth = None
        self.depth_initialized = False
        self.depth_threshold = 0.1  # 深度检测阈值
        self.target_distance = 1.0  # 期望距离

    def depth_callback(self, msg):
        """处理深度图像"""
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, "32FC1")
            if trackObject == 1 and 'track_window' in globals():
                # 获取跟踪窗口区域的深度
                x, y = int(track_window[0] + track_window[2]/2), int(track_window[1] + track_window[3]/2)
                window_depth = self.depth_image[y-5:y+5, x-5:x+5]  # 使用5x5区域的平均深度
                valid_depths = window_depth[~np.isnan(window_depth)]
                
                if valid_depths.size > 0:
                    self.current_depth = np.median(valid_depths)  # 使用中值滤波
                    if not self.depth_initialized and not np.isnan(self.current_depth):
                        self.target_depth = self.current_depth
                        self.depth_initialized = True
                    rospy.loginfo("深度信息 - 当前: %.2f m, 目标: %.2f m", 
                                self.current_depth, self.target_depth)

        except Exception as e:
            rospy.logerr("深度处理失败: %s", str(e))

    def calculate_control(self, target_x, length):
        """结合视觉和深度信息计算控制命令"""
        msg = Twist()
        center_x = 320
        error_x = target_x - center_x
        
        # 方向控制
        if abs(error_x) > self.threshold:
            turn_factor = min(abs(error_x) / center_x, 1.0)
            msg.angular.z = self.angular_z * turn_factor * (error_x / abs(error_x))
        
        # 深度控制
        if hasattr(self, 'current_depth') and self.depth_initialized:
            depth_error = self.current_depth - self.target_distance
            rospy.loginfo("深度误差: %.2f m", depth_error)
            
            if abs(depth_error) > self.depth_threshold:
                # 根据深度误差计算速度
                speed = self.linear_x * np.clip(depth_error, -1, 1)
                msg.linear.x = speed
                
                # 距离太近时后退
                if depth_error < -self.depth_threshold:
                    msg.linear.x = -abs(msg.linear.x)
                # 距离太远时前进
                elif depth_error > self.depth_threshold:
                    msg.linear.x = abs(msg.linear.x)
            else:
                msg.linear.x = 0  # 距离合适时停止
                
            # 转向时降低速度
            msg.linear.x *= (1.0 - 0.5 * abs(msg.angular.z))
        else:
            # 如果没有深度信息，使用视觉大小估计
            if length < self.track_windows_threshold:
                msg.linear.x = self.linear_x
            else:
                msg.linear.x = 0
        
        rospy.loginfo("控制输出 - 线速度: %.2f, 角速度: %.2f", msg.linear.x, msg.angular.z)
        return msg

    def image_sub_callback(self, msg):
        global image
        try:
            image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # 显示深度信息
            if self.depth_image is not None:
                depth_display = cv2.normalize(self.depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                depth_colormap = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
                cv2.imshow('depth', depth_colormap)
            
            track_centerX, length_of_diagonal = ExamByCamshift()
            windows_centerX = image.shape[1] // 2  # 使用图像实际中心点
            
            if track_centerX > 0:  # 只在有效跟踪时控制
                error = track_centerX - windows_centerX
                rospy.loginfo("控制 - 中心: %.1f, 误差: %.1f, 阈值: %d", 
                            track_centerX, error, self.threshold)
                
                if abs(error) > self.threshold:
                    if error < 0:
                        self.turn_right()
                    else:
                        self.turn_left()
                else:
                    if length_of_diagonal < self.track_windows_threshold:
                        self.go_ahead()
                    else:
                        self.stop_move()
            
        except Exception as e:
            rospy.logerr("图像处理失败: %s", str(e))

    def turn_left(self):
        rospy.loginfo("cam_turn_left")
        msg = Twist()
        msg.angular.z = -self.angular_z
        self.twist_pub.publish(msg)
    def turn_right(self):
        rospy.loginfo("cam_turn_right")
        msg = Twist()
        msg.angular.z = self.angular_z
        self.twist_pub.publish(msg)
    def stop_move(self):
        """停止移动"""
        if hasattr(self, 'twist_pub') and self.twist_pub is not None:
            rospy.loginfo("find_target")
            msg = Twist()
            self.twist_pub.publish(msg)
    def go_ahead(self):
        rospy.loginfo("moving ahead")
        msg = Twist()
        msg.linear.x = -self.linear_x
        self.twist_pub.publish(msg)
    def go_back(self):
        rospy.loginfo("moving back")
        msg = Twist()
        msg.linear.x = self.linear_x
        self.twist_pub.publish(msg)

def callback(twist):
    print (twist.linear.x)
    print(1)
if __name__ == '__main__':
    try:
        rospy.init_node('image_listenner', anonymous=False)
        image_listenning = image_listenner()
        rospy.on_shutdown(cv2.destroyAllWindows)
        rospy.spin()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
    except Exception as e:
        rospy.logerr("Node error: %s", str(e))
    finally:
        cv2.destroyAllWindows()
        cv2.waitKey(1)
