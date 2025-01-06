#!/usr/bin/env python
# -*- coding: utf-8 -*-
#############
# 视觉跟踪节点：通过OpenCV进行目标选择和方向计算
#############
import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import numpy as np
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
    if selectObject == True:
        xs = min(x, xo)
        ys = min(y, yo)
        ws = abs(x - xo)
        hs = abs(y - yo)
    if event == cv2.EVENT_LBUTTONDOWN:
        xo, yo = x, y
        xs, ys, ws, hs = x, y, 0, 0
        selectObject = True
    elif event == cv2.EVENT_LBUTTONUP:
        selectObject = False
        trackObject = -1
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
    if trackObject != 0:
        mask = cv2.inRange(hsv, np.array((0., 30., 10.)), np.array((180., 256., 255.)))
        if trackObject == -1:
            track_window = (xs, ys, ws, hs)
            maskroi = mask[ys:ys + hs, xs:xs + ws]
            hsv_roi = hsv[ys:ys + hs, xs:xs + ws]
            roi_hist = cv2.calcHist([hsv_roi], [0], maskroi, [180], [0, 180])
            cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
            trackObject = 1
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        dst &= mask
        ret, track_window = cv2.CamShift(dst, track_window, term_crit)
        centerX = ret[0][0]
        #data : ret (center(x,y),(width,height),angular)
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        length_of_diagonal = math.sqrt(ret[1][1] ** 2 + ret[1][0] ** 2)
        img2 = cv2.polylines(image, [pts], True, 255, 2)
    if selectObject == True and ws > 0 and hs > 0:
        #cv2.imshow('imshow1', image[ys:ys + hs, xs:xs + ws])
        cv2.bitwise_not(image[ys:ys + hs, xs:xs + ws], image[ys:ys + hs, xs:xs + ws])
    cv2.imshow('imshow', image)

    return centerX, length_of_diagonal

class image_listenner:
    def __init__(self):
        # 基本参数设置
        self.threshold = 80  # 降低阈值提高灵敏度
        self.linear_x = 0.3  # 降低速度使运动更平滑
        self.angular_z = 0.15
        self.track_windows_threshold = math.sqrt(95*95+235*235)+10000
        self.target_distance = 1.2  # 增加跟随距离
        self.distance_threshold = 0.2
        self.prev_centerX = 320  # 添加目标位置历史记录
        self.smooth_factor = 0.3  # 平滑因子
        
        try:
            # 修改窗口创建方式
            cv2.namedWindow('imshow', cv2.WINDOW_AUTOSIZE)
            cv2.moveWindow('imshow', 100, 100)
            cv2.setMouseCallback('imshow', onMouse)
            rospy.loginfo("Initializing image viewer...")
            
            # 修改订阅话题
            self.bridge = CvBridge()
            self.image_sub = rospy.Subscriber("/track_car/camera/image_raw", 
                                            Image, 
                                            self.image_sub_callback,
                                            queue_size=1,
                                            buff_size=2**24)  # 增加缓冲区大小
            self.depth_sub = rospy.Subscriber("/camera/depth/image_raw", 
                                            Image, 
                                            self.depth_callback,
                                            queue_size=1)
            self.twist_pub = rospy.Publisher("cmd_vel", Twist, queue_size=1)
            
            # 等待第一帧图像
            rospy.wait_for_message("/track_car/camera/image_raw", Image, timeout=5.0)
            
        except Exception as e:
            rospy.logerr("Init failed: %s", str(e))
            raise

        self.start_auto_detect()

    def __del__(self):
        try:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        except:
            pass

    def start_auto_detect(self):
        """自动开始目标检测"""
        global trackObject, xs, ys, ws, hs
        rospy.sleep(2)  # 等待图像稳定
        
        # 在图像中心设置初始检测窗口
        xs, ys = 220, 180
        ws, hs = 200, 150
        trackObject = -1
        
        rospy.loginfo("目标检测窗口已启动")
        rospy.loginfo("请点击并拖动鼠标来选择跟踪目标")

    def predict_target_position(self, current_x):
        """预测目标位置"""
        predicted_x = current_x + (current_x - self.prev_centerX) * self.smooth_factor
        self.prev_centerX = current_x
        return predicted_x

    def calculate_control(self, target_x, depth):
        """计算更平滑的控制输出"""
        msg = Twist()
        windows_centerX = 320
        
        # 方向控制
        error_x = target_x - windows_centerX
        msg.angular.z = self.angular_z * (error_x / windows_centerX)
        
        # 速度控制
        if depth > 0:
            error_distance = depth - self.target_distance
            speed_factor = min(abs(error_distance), 1.0)
            if error_distance > self.distance_threshold:
                msg.linear.x = -self.linear_x * speed_factor
            elif error_distance < -self.distance_threshold:
                msg.linear.x = self.linear_x * speed_factor
        
        return msg

    def depth_callback(self, msg):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(msg, "32FC1")
            # 获取跟踪窗口中心点的深度值
            if trackObject == 1:
                x, y = int(track_window[0] + track_window[2]/2), int(track_window[1] + track_window[3]/2)
                self.current_depth = depth_image[y, x]
                if math.isnan(self.current_depth):
                    self.current_depth = 0
        except:
            rospy.logerr("depth image processing failed")

    def image_sub_callback(self, msg):
        '''图像订阅回调函数'''
        global image

        try:
            # 转换和显示图像
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            image = cv_image.copy()
            
            if not trackObject:
                cv2.putText(image, "Click and drag to select target", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                          0.7, (0, 255, 0), 2)
                
            track_centerX, length_of_diagonal = ExamByCamshift()
            
            if track_centerX >= 0:
                predicted_x = self.predict_target_position(track_centerX)
                control_msg = self.calculate_control(predicted_x, self.current_depth)
                self.twist_pub.publish(control_msg)
            
            cv2.imshow('imshow', image)
            cv2.waitKey(3)
            
        except Exception as e:
            rospy.logerr("Image processing failed: %s", str(e))

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
