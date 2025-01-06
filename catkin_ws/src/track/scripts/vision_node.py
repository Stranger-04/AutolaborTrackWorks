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
    
    # 添加调试信息
    print_debug = True  # 用于调试的开关
    centerX = -1.0
    length_of_diagonal = float()
    
    try:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 处理框选
        if selectObject and ws > 0 and hs > 0:
            cv2.bitwise_not(image[ys:ys + hs, xs:xs + ws], image[ys:ys + hs, xs:xs + ws])
            if print_debug:
                rospy.loginfo("框选区域: x=%d, y=%d, w=%d, h=%d", xs, ys, ws, hs)
        
        # 跟踪处理
        if trackObject != 0:
            mask = cv2.inRange(hsv, np.array((0., 30., 10.)), np.array((180., 256., 255.)))
            
            if trackObject == -1:  # 初始化跟踪
                track_window = (xs, ys, ws, hs)
                maskroi = mask[ys:ys + hs, xs:xs + ws]
                hsv_roi = hsv[ys:ys + hs, xs:xs + ws]
                roi_hist = cv2.calcHist([hsv_roi], [0], maskroi, [180], [0, 180])
                cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
                trackObject = 1
                if print_debug:
                    rospy.loginfo("初始化跟踪窗口: (%d,%d) %dx%d", xs, ys, ws, hs)
            
            dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
            dst &= mask
            
            term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
            ret, track_window = cv2.CamShift(dst, track_window, term_crit)
            centerX = float(ret[0][0])
            
            # 添加跟踪结果调试信息
            if print_debug:
                rospy.loginfo("跟踪结果 - Center: (%.1f, %.1f), Box: %dx%d", 
                            ret[0][0], ret[0][1], ret[1][0], ret[1][1])
            
            pts = cv2.boxPoints(ret)
            pts = np.int0(pts)
            length_of_diagonal = math.sqrt(float(ret[1][1]) ** 2 + float(ret[1][0]) ** 2)
            cv2.polylines(image, [pts], True, (0, 0, 255), 2)
            
            # 显示中心点和跟踪框
            cv2.circle(image, (int(centerX), int(ret[0][1])), 5, (0, 255, 255), -1)
            cv2.putText(image, "Center: ({:.0f}, {:.0f})".format(centerX, ret[0][1]),
                       (int(centerX)-50, int(ret[0][1])-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        cv2.imshow('imshow', image)
        cv2.waitKey(3)
        
    except Exception as e:
        rospy.logerr("跟踪失败: %s", str(e))
        return -1.0, 0.0
        
    return centerX, length_of_diagonal

class image_listenner:
    def __init__(self):
        # 调整控制参数
        self.threshold = 50  # 降低阈值使转向不那么敏感
        self.linear_x = 0.2  # 降低速度
        self.angular_z = 0.1  # 降低转向速度
        self.track_windows_threshold = math.sqrt(95*95+235*235)+10000
        
        # 简化初始化
        cv2.namedWindow('imshow', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('imshow', onMouse)
        self.bridge = CvBridge()
        
        # 修正话题名
        self.image_sub = rospy.Subscriber("/track_car/camera/image_raw", 
                                        Image, 
                                        self.image_sub_callback)
        self.twist_pub = rospy.Publisher("cmd_vel", Twist, queue_size=10)
    
    def image_sub_callback(self, msg):
        global image
        try:
            image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            track_centerX, length_of_diagonal = ExamByCamshift()
            windows_centerX = 320
            
            if track_centerX >= 0:
                error = track_centerX - windows_centerX
                
                # 添加调试信息
                rospy.loginfo("Control - Center: %.1f, Error: %.1f, Threshold: %d", 
                            track_centerX, error, self.threshold)
                
                # 修改控制逻辑
                if abs(error) > self.threshold:
                    if error < 0:  # 目标在左边
                        self.turn_right()
                    else:  # 目标在右边
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
