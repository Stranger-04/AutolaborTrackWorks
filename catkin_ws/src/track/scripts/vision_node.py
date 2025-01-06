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
    # 创建显示用的图像副本
    display_img = image.copy()
    
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    centerX = -1.0
    length_of_diagonal = float()
    
    # 修正rectangle参数
    if selectObject and ws > 0 and hs > 0:
        cv2.rectangle(display_img, (int(xs), int(ys)), 
                     (int(xs+ws), int(ys+hs)), (0, 255, 0), 2)
        cv2.putText(display_img, "Selection: {}x{}".format(int(ws), int(hs)), 
                    (int(xs), int(ys-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # 目标跟踪处理
    if trackObject != 0:
        try:
            mask = cv2.inRange(hsv, np.array((0., 30., 10.)), np.array((180., 256., 255.)))
            if trackObject == -1:  # 初次选择目标
                track_window = (int(xs), int(ys), int(ws), int(hs))
                maskroi = mask[ys:ys + hs, xs:xs + ws]
                hsv_roi = hsv[ys:ys + hs, xs:xs + ws]
                roi_hist = cv2.calcHist([hsv_roi], [0], maskroi, [180], [0, 180])
                cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
                trackObject = 1
                
                # 显示选中区域
                cv2.rectangle(display_img, (int(xs), int(ys)), 
                            (int(xs+ws), int(ys+hs)), (255, 0, 0), 2)
                cv2.putText(display_img, "Target Selected", (int(xs), int(ys-10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # 执行跟踪
            dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
            dst &= mask
            ret, track_window = cv2.CamShift(dst, track_window, term_crit)
            
            # 确保返回值为整数
            centerX = float(ret[0][0])
            pts = cv2.boxPoints(ret)
            pts = np.int0(pts)
            length_of_diagonal = math.sqrt(float(ret[1][1]) ** 2 + float(ret[1][0]) ** 2)
            
            # 显示跟踪框和中心点
            cv2.polylines(display_img, [pts], True, (0, 0, 255), 2)
            cv2.circle(display_img, (int(centerX), int(ret[0][1])), 5, (0, 255, 255), -1)
            cv2.putText(display_img, "Tracking: ({}, {})".format(int(centerX), int(ret[0][1])), 
                        (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        except Exception as e:
            rospy.logerr("跟踪处理失败: %s", str(e))
            trackObject = 0
    
    cv2.imshow('imshow', display_img)
    return centerX, length_of_diagonal

class image_listenner:
    def __init__(self):
        # 初始化Publisher
        self.twist_pub = None  # 将在初始化完成后创建
        
        # 调整控制参数
        self.threshold = 80
        self.linear_x = 0.2  # 降低线速度使运动更稳定
        self.angular_z = 0.3  # 增加角速度提高转向灵敏度
        self.target_distance = 1.0  # 目标跟随距离
        self.distance_threshold = 0.1  # 降低阈值提高精度
        self.prev_centerX = 320
        self.smooth_factor = 0.5  # 增加平滑因子提高响应性
        self.current_depth = 0
        
        try:
            # 初始化ROS节点和参数
            rospy.loginfo("等待相机初始化...")
            
            # 修改窗口创建方式
            cv2.namedWindow('imshow', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('imshow', 800, 600)  # 放大显示窗口
            cv2.moveWindow('imshow', 100, 100)
            cv2.setMouseCallback('imshow', onMouse)
            
            # 初始化bridge和订阅器
            self.bridge = CvBridge()
            
            # 确保使用正确的话题名
            camera_topic = "camera/image_raw"  # 相对命名空间
            rospy.loginfo("订阅相机话题: %s", camera_topic)
            
            # 订阅相机图像，增加超时重试机制
            retry_count = 0
            while not rospy.is_shutdown() and retry_count < 3:
                try:
                    self.image_sub = rospy.Subscriber(camera_topic, 
                                                    Image, 
                                                    self.image_sub_callback,
                                                    queue_size=1,
                                                    buff_size=2**24)
                    
                    # 等待第一帧图像
                    rospy.wait_for_message(camera_topic, Image, timeout=2.0)
                    rospy.loginfo("成功接收到相机图像")
                    break
                except Exception as e:
                    retry_count += 1
                    rospy.logwarn("等待相机图像超时，重试 %d/3...", retry_count)
                    rospy.sleep(1)
            
            if retry_count >= 3:
                raise Exception("无法接收相机图像，请检查相机话题是否正确")
            
            # 其他订阅器初始化
            self.depth_sub = rospy.Subscriber("camera/depth/image_raw", 
                                            Image, 
                                            self.depth_callback,
                                            queue_size=1)
            self.twist_pub = rospy.Publisher("cmd_vel", Twist, queue_size=1)
            
        except Exception as e:
            rospy.logerr("初始化失败: %s", str(e))
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
        
        # 方向控制 - 基于目标在图像中的位置
        error_x = target_x - windows_centerX
        # 增加非线性控制以提高响应性
        if abs(error_x) > 100:  # 大偏差时快速转向
            msg.angular.z = self.angular_z * 1.5 * (error_x / windows_centerX)
        else:  # 小偏差时精细调整
            msg.angular.z = self.angular_z * (error_x / windows_centerX)
        
        # 速度控制 - 基于深度信息
        if depth > 0:
            error_distance = depth - self.target_distance
            speed_factor = min(abs(error_distance), 1.0)
            
            if abs(error_distance) > self.distance_threshold:
                if error_distance > 0:  # 目标太远
                    msg.linear.x = self.linear_x * speed_factor
                else:  # 目标太近
                    msg.linear.x = -self.linear_x * speed_factor
            else:  # 在合适距离内停止
                msg.linear.x = 0.0
                
            # 当转向时降低前进速度
            msg.linear.x *= (1 - abs(msg.angular.z))
            
            rospy.loginfo("Control: distance={:.2f}, speed={:.2f}, turn={:.2f}".format(
                         depth, msg.linear.x, msg.angular.z))
        
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
            
            # 添加状态和调试信息显示
            status_text = "选择目标" if not trackObject else "跟踪中"
            cv2.putText(image, "状态: {}".format(status_text), (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 处理跟踪
            track_centerX, length_of_diagonal = ExamByCamshift()
            
            if track_centerX >= 0 and trackObject == 1:  # 确保处于跟踪状态
                predicted_x = self.predict_target_position(track_centerX)
                control_msg = self.calculate_control(predicted_x, self.current_depth)
                
                # 添加调试信息显示
                cv2.putText(image, "Target X: {:.1f}, Depth: {:.2f}m".format(
                           predicted_x, self.current_depth), (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # 发布控制命令
                self.twist_pub.publish(control_msg)
            else:
                # 未跟踪时停止移动
                self.stop_move()
            
            # 确保显示更新
            cv2.waitKey(1)
            
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
