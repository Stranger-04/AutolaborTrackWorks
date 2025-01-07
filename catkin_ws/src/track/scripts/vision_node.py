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
    centerX = -1.0
    length_of_diagonal = float()

    try:
        # 转换为HSV空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 处理框选过程
        if selectObject and ws > 0 and hs > 0:
            # 修正：确保所有坐标都是整数
            x1, y1 = int(xs), int(ys)
            x2, y2 = int(xs + ws), int(ys + hs)
            
            # 绘制选择框
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 反色显示选择区域
            try:
                selection = image[y1:y2, x1:x2]
                if selection.size > 0:  # 确保选择区域有效
                    cv2.bitwise_not(selection, selection)
            except ValueError as e:
                rospy.logwarn("Invalid selection area: %s", str(e))
            
            rospy.loginfo("选择区域: x1=%d, y1=%d, w=%d, h=%d", x1, y1, ws, hs)

        # 跟踪处理
        if trackObject != 0:
            # 设置HSV掩码范围
            mask = cv2.inRange(hsv, np.array((0., 30., 10.)), np.array((180., 256., 255.)))
            
            if trackObject == -1:  # 初始化跟踪
                if ws > 0 and hs > 0:
                    # 保存跟踪窗口
                    track_window = (int(xs), int(ys), int(ws), int(hs))
                    
                    # 确保索引有效
                    y1, y2 = int(ys), int(ys + hs)
                    x1, x2 = int(xs), int(xs + ws)
                    
                    # 确保索引在图像范围内
                    y1 = max(0, min(y1, image.shape[0] - 1))
                    y2 = max(0, min(y2, image.shape[0]))
                    x1 = max(0, min(x1, image.shape[1] - 1))
                    x2 = max(0, min(x2, image.shape[1]))
                    
                    maskroi = mask[y1:y2, x1:x2]
                    hsv_roi = hsv[y1:y2, x1:x2]
                    
                    # 检查ROI区域是否有效
                    if maskroi.size > 0 and hsv_roi.size > 0:
                        roi_hist = cv2.calcHist([hsv_roi], [0], maskroi, [180], [0, 180])
                        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
                        trackObject = 1
                        rospy.loginfo("跟踪初始化成功")
                    else:
                        rospy.logerr("无效的ROI区域")
                        trackObject = 0
            
            if trackObject == 1:
                # 执行反向投影和跟踪
                dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
                dst &= mask
                
                ret, track_window = cv2.CamShift(dst, track_window, term_crit)
                
                centerX = float(ret[0][0])
                pts = cv2.boxPoints(ret)
                pts = np.int0(pts)
                length_of_diagonal = math.sqrt(float(ret[1][1]) ** 2 + float(ret[1][0]) ** 2)
                
                # 绘制跟踪结果
                cv2.polylines(image, [pts], True, (0, 0, 255), 2)
                cv2.circle(image, (int(centerX), int(ret[0][1])), 5, (0, 255, 255), -1)
                
                # 显示跟踪信息
                cv2.putText(image, "Tracking: ({:.0f}, {:.0f})".format(centerX, ret[0][1]),
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

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
        self.linear_x = 0.2  # 降低基础速度以提高稳定性
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
        self.depth_threshold = 0.05  # 深度检测阈值
        self.target_distance = 1.65  # 期望距离
        self.max_speed = 0.3  # 最大速度限制

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
                # 使用比例控制计算速度
                speed_factor = np.clip(abs(depth_error) / 0.5, 0, 1.0)  # 0.5m作为标准差
                speed = self.linear_x * speed_factor
                
                # 确保速度不超过最大限制
                speed = np.clip(speed, -self.max_speed, self.max_speed)
                
                # 根据距离差决定前进还是后退
                if depth_error > 0:  # 当前距离大于目标距离，需要前进
                    msg.linear.x = speed
                else:  # 当前距离小于目标距离，需要后退
                    msg.linear.x = -speed
                
                rospy.loginfo("深度控制 - 当前距离: %.2f m, 目标距离: %.2f m, 速度: %.2f", 
                            self.current_depth, self.target_distance, msg.linear.x)
            else:
                msg.linear.x = 0  # 在目标范围内停止
                rospy.loginfo("到达目标距离范围")
            
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
                
                # 首先处理方向控制
                if abs(error) > self.threshold:
                    if error < 0:
                        self.turn_right()
                    else:
                        self.turn_left()
                else:
                    # 方向正确时，根据深度信息控制前进后退
                    if hasattr(self, 'current_depth') and self.depth_initialized:
                        depth_error = self.current_depth - self.target_distance
                        if abs(depth_error) > self.depth_threshold:
                            if depth_error > 0:  # 当前距离大于目标距离，需要前进靠近
                                self.go_ahead()
                            else:  # 当前距离小于目标距离，需要后退
                                self.go_back()
                        else:
                            self.stop_move()  # 在目标范围内停止
                    else:
                        # 没有深度信息时使用视觉大小估计
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
        """前进"""
        rospy.loginfo("moving ahead")
        msg = Twist()
        msg.linear.x = self.linear_x  # 注意：这里移除了负号
        self.twist_pub.publish(msg)
    def go_back(self):
        """后退"""
        rospy.loginfo("moving back")
        msg = Twist()
        msg.linear.x = -self.linear_x  # 加上负号表示后退
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
