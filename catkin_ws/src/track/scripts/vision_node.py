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
    display_img = image.copy()
    centerX = -1.0  # 初始化返回值
    length_of_diagonal = 0.0  # 初始化返回值
    
    # 定义跟踪参数
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    
    try:
        # 显示深度信息
        if hasattr(image_listenning, 'depth_image') and image_listenning.depth_image is not None:
            depth_display = cv2.normalize(image_listenning.depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_color = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
            cv2.imshow('depth', depth_color)
            
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 初始化深度相关变量
        depth_mask = None
        mask = None
        
        # 设置基础HSV范围
        lower_bound = np.array((0., 20., 30.))
        upper_bound = np.array((180., 255., 255.))
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        if selectObject and ws > 0 and hs > 0:
            # 显示选择框
            cv2.rectangle(display_img, (xs, ys), (xs+ws, ys+hs), (0, 255, 0), 2)
            
            # 如果有深度信息，显示选择区域的平均深度
            if hasattr(image_listenning, 'depth_image') and image_listenning.depth_image is not None:
                roi_depth = image_listenning.depth_image[ys:ys+hs, xs:xs+ws]
                mean_depth = np.nanmean(roi_depth)
                cv2.putText(display_img, "Depth: {:.2f}m".format(mean_depth), 
                           (xs, ys-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 目标跟踪处理
        if trackObject != 0:
            # 创建深度掩码
            depth_mask = np.ones_like(mask, dtype=np.uint8) * 255
            if hasattr(image_listenning, 'depth_image') and image_listenning.depth_image is not None:
                # 使用深度范围过滤
                target_depth = image_listenning.target_depth
                depth_tolerance = 0.5  # 深度容差范围
                depth_mask = cv2.inRange(
                    image_listenning.depth_image,
                    target_depth - depth_tolerance,
                    target_depth + depth_tolerance
                )
            
            # 结合HSV和深度掩码
            if trackObject == -1:
                track_window = (int(xs), int(ys), int(ws), int(hs))
                maskroi = mask[ys:ys + hs, xs:xs + ws]
                hsv_roi = hsv[ys:ys + hs, xs:xs + ws]
                
                # 存储目标深度
                if hasattr(image_listenning, 'depth_image') and image_listenning.depth_image is not None:
                    roi_depth = image_listenning.depth_image[ys:ys+hs, xs:xs+ws]
                    image_listenning.target_depth = np.nanmean(roi_depth)
                
                # 记录目标的HSV范围
                rospy.loginfo("目标HSV范围 - H: %d-%d, S: %d-%d, V: %d-%d",
                            np.min(hsv_roi[:,:,0]), np.max(hsv_roi[:,:,0]),
                            np.min(hsv_roi[:,:,1]), np.max(hsv_roi[:,:,1]),
                            np.min(hsv_roi[:,:,2]), np.max(hsv_roi[:,:,2]))
                
                # 使用目标实际的HSV范围创建掩码
                target_lower = np.array((
                    max(0, np.min(hsv_roi[:,:,0])-10),
                    max(0, np.min(hsv_roi[:,:,1])-40),
                    max(0, np.min(hsv_roi[:,:,2])-40)
                ))
                target_upper = np.array((
                    min(180, np.max(hsv_roi[:,:,0])+10),
                    min(255, np.max(hsv_roi[:,:,1])+40),
                    min(255, np.max(hsv_roi[:,:,2])+40)
                ))
                
                # 重新计算掩码
                mask = cv2.inRange(hsv, target_lower, target_upper)
                maskroi = mask[ys:ys + hs, xs:xs + ws]
                
                # 计算直方图
                roi_hist = cv2.calcHist([hsv_roi], [0, 1, 2], maskroi, [32, 32, 32], 
                                      [0, 180, 0, 256, 0, 256])
                cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
                trackObject = 1
                
            # 改进的目标跟踪
            mask = cv2.inRange(hsv, lower_bound, upper_bound)
            mask = cv2.bitwise_and(mask, depth_mask)
            
            dst = cv2.calcBackProject([hsv], [0, 1, 2], roi_hist, [0, 180, 0, 256, 0, 256], 1)
            dst = cv2.bitwise_and(dst, mask)
            
            # 添加形态学处理
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
            dst = cv2.filter2D(dst, -1, kernel)
            dst = cv2.threshold(dst, 50, 255, cv2.THRESH_BINARY)[1]
            dst = cv2.erode(dst, None, iterations=2)
            dst = cv2.dilate(dst, None, iterations=2)
            
            # CamShift跟踪
            ret, track_window = cv2.CamShift(dst, track_window, term_crit)
            
            # 更新跟踪结果
            pts = cv2.boxPoints(ret)
            pts = np.int0(pts)
            centerX = float(ret[0][0])
            length_of_diagonal = math.sqrt(float(ret[1][1]) ** 2 + float(ret[1][0]) ** 2)
            
            # 绘制跟踪结果
            cv2.polylines(display_img, [pts], True, (0, 0, 255), 2)
            cv2.circle(display_img, (int(centerX), int(ret[0][1])), 5, (0, 255, 255), -1)
            
            # 显示深度信息
            if hasattr(image_listenning, 'depth_image'):
                current_depth = image_listenning.current_depth
                cv2.putText(display_img, "Current Depth: {:.2f}m".format(current_depth),
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow('imshow', display_img)
        cv2.waitKey(1)
        
    except Exception as e:
        rospy.logerr("跟踪处理失败: %s", str(e))
        return -1.0, 0.0
        
    return centerX, length_of_diagonal

class image_listenner:
    def __init__(self):
        # 初始化Publisher
        self.twist_pub = None  # 将在初始化完成后创建
        
        # 调整控制参数
        self.threshold = 50  # 降低阈值，提高响应性
        self.linear_x = 0.2  # 降低速度使运动更平滑
        self.angular_z = 0.3
        self.distance_factor = 0.001  # 距离系数
        self.target_size = 150  # 目标大小（像素）
        self.min_size = 50  # 最小目标大小
        self.max_size = 250  # 最大目标大小
        
        # 控制阈值
        self.center_threshold = 30  # 中心点偏差阈值
        self.size_threshold = 20  # 大小偏差阈值
        self.target_distance = 1.0  # 目标跟随距离
        self.distance_threshold = 0.1  # 降低阈值提高精度
        self.prev_centerX = 320
        self.smooth_factor = 0.5  # 增加平滑因子提高响应性
        self.current_depth = 0
        
        # 跟踪相关参数
        self.track_lost_count = 0
        self.max_track_lost = 10  # 降低最大丢失帧数
        self.min_target_size = 30  # 降低最小目标大小阈值
        
        try:
            # 初始化ROS节点和参数
            rospy.loginfo("等待相机初始化...")
            
            # 修改窗口创建方式
            cv2.namedWindow('imshow', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('imshow', 800, 600)  # 放大显示窗口
            cv2.moveWindow('imshow', 100, 100)
            cv2.setMouseCallback('imshow', onMouse)
            rospy.loginfo("创建窗口成功，等待图像...")
            
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
            
            rospy.loginfo("初始化完成，按R键重置跟踪，Q键退出")
            
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

    def calculate_control(self, target_x, length):
        """计算控制命令"""
        msg = Twist()
        center_x = 320  # 图像中心x坐标
        
        # 计算水平偏差
        error_x = target_x - center_x
        
        # 方向控制
        if abs(error_x) > self.center_threshold:
            # 根据偏差大小非线性调整转向速度
            turn_factor = min(abs(error_x) / center_x, 1.0)
            msg.angular.z = self.angular_z * turn_factor * (error_x / abs(error_x))
        
        # 距离控制
        if length > 0:
            # 使用目标大小估算距离
            size_error = length - self.target_size
            
            if abs(size_error) > self.size_threshold:
                # 根据目标大小决定前进后退
                if size_error < 0:  # 目标太小，需要前进
                    msg.linear.x = self.linear_x
                else:  # 目标太大，需要后退
                    msg.linear.x = -self.linear_x
                
                # 根据偏差大小调整速度
                speed_factor = min(abs(size_error) / self.target_size, 1.0)
                msg.linear.x *= speed_factor
            
            # 当转向时降低速度
            msg.linear.x *= (1.0 - 0.5 * abs(msg.angular.z))
            
            rospy.loginfo("Control - Error_x: %.2f, Size: %.2f, Speed: %.2f, Turn: %.2f" % 
                         (error_x, length, msg.linear.x, msg.angular.z))
        
        # 使用深度信息进行更精确的控制
        if self.current_depth is not None and self.target_depth is not None:
            depth_error = self.current_depth - self.target_depth
            
            # 距离控制
            if abs(depth_error) > self.distance_threshold:
                speed = self.linear_x * np.clip(depth_error, -1, 1)
                msg.linear.x = speed
            
            # 根据深度调整转向速度
            distance_factor = min(self.current_depth / 2.0, 1.0)  # 距离越远，转向越慢
            error_x = target_x - 320
            if abs(error_x) > self.center_threshold:
                msg.angular.z = self.angular_z * (error_x / 320.0) * distance_factor
        
        return msg

    def depth_callback(self, msg):
        global track_window
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, "32FC1")
            if trackObject == 1 and 'track_window' in globals():
                x, y = int(track_window[0] + track_window[2]/2), int(track_window[1] + track_window[3]/2)
                if 0 <= y < self.depth_image.shape[0] and 0 <= x < self.depth_image.shape[1]:
                    depth = self.depth_image[y, x]
                    if not math.isnan(depth):
                        self.current_depth = depth
                    
                    # 使用深度信息调整跟踪参数
                    if self.target_depth is not None:
                        depth_error = depth - self.target_depth
                        # 根据深度差异调整跟踪窗口大小
                        scale = self.target_depth / depth if depth > 0 else 1.0
                        track_window = (
                            track_window[0],
                            track_window[1],
                            int(track_window[2] * scale),
                            int(track_window[3] * scale)
                        )
        except Exception as e:
            rospy.logerr("深度图处理失败: %s", str(e))

    def image_sub_callback(self, msg):
        '''图像订阅回调函数'''
        global image, trackObject

        try:
            # 转换和显示图像
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            if cv_image is None:
                rospy.logerr("接收到空图像")
                return
                
            image = cv_image.copy()
            
            # 添加状态和调试信息显示
            status_text = "选择目标" if not trackObject else "跟踪中"
            cv2.putText(image, "状态: {}".format(status_text), (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 处理跟踪
            track_centerX, length_of_diagonal = ExamByCamshift()
            
            # 添加图像接收和处理状态日志
            rospy.logdebug("图像大小: %dx%d" % (image.shape[1], image.shape[0]))
            
            if track_centerX >= 0 and length_of_diagonal > self.min_target_size:
                self.track_lost_count = 0  # 重置丢失计数
                predicted_x = self.predict_target_position(track_centerX)
                control_msg = self.calculate_control(predicted_x, length_of_diagonal)
                
                # 添加调试信息显示
                cv2.putText(image, "Target X: %.1f, Size: %.1f" % 
                           (predicted_x, length_of_diagonal), (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(image, "Control - Turn: %.2f, Speed: %.2f" % 
                           (control_msg.angular.z, control_msg.linear.x), (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # 发布控制命令
                self.twist_pub.publish(control_msg)
            else:
                self.track_lost_count += 1
                if self.track_lost_count > self.max_track_lost:
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
