#!/usr/bin/env python
#############
#
#Node :choose target by opencv and calculate the direction
#
#############
import rospy
import cv_bridge
from sensor_msgs.msg import Image
import cv2
import numpy as np
import math
from geometry_msgs.msg import Twist
xs, ys, ws, hs = 0, 0, 0, 0  # selection.x selection.y
xo, yo = 0, 0  # origin.x origin.y
selectObject = False
trackObject = 0
#################
#
#   todos:
#       choose a area as a target
#
#################
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
#################
#
#   todos:
#       track the target area by camshift
#
#################
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
        
        # 初始化bridge和订阅器
        self.bridge = cv_bridge.CvBridge()
        self.image_sub = rospy.Subscriber("/usb_cam/image_raw", Image, self.image_sub_callback)
        self.depth_sub = rospy.Subscriber("/camera/depth/image_raw", Image, self.depth_callback)
        self.twist_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        
        self.current_depth = 0
        self.start_auto_detect()  # 添加自动检测启动

    def start_auto_detect(self):
        """自动开始目标检测"""
        global trackObject, xs, ys, ws, hs
        # 设置初始检测窗口在图像中心
        rospy.sleep(2)  # 等待图像稳定
        xs, ys = 220, 180  # 设置初始检测窗口位置
        ws, hs = 200, 150  # 设置初始检测窗口大小
        trackObject = -1   # 触发跟踪初始化

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
        ''' callback of image_sub '''
        global image

        try:
            self.img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            image = self.img
            track_centerX, length_of_diagonal = ExamByCamshift()
            
            if track_centerX >= 0:
                # 使用预测位置进行控制
                predicted_x = self.predict_target_position(track_centerX)
                control_msg = self.calculate_control(predicted_x, self.current_depth)
                self.twist_pub.publish(control_msg)
            else:
                rospy.loginfo("Finding target...")
                
            cv2.setMouseCallback('imshow', onMouse)
            cv2.waitKey(3)
        except:
            rospy.logerr("img get failed")


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
    rospy.init_node('image_listenner', anonymous=False)
    # rospy.Subscriber("/cmd_vel",Twist,callback)
    # while(True):
    #     rospy.spin()
    # rate = 50
    # r = rospy.Rate(rate)
    #
    # msg = Twist()
    # msg.linear.x = 1
    # msg.linear.y = 0
    # msg.linear.z = 0
    # msg.angular.x = 0
    # msg.angular.y = 0
    # msg.angular.z = 0
    # pub = rospy.Publisher("/cmd_vel",Twist,queue_size=1)
    # while(True):
    #     pub.publish(msg)
    #     r.sleep()
    image_listenning = image_listenner()
    #movebase = MoveBase()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
