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
    # 创建显示用的图像副本
    display_img = image.copy()
    
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    centerX = -1.0
    length_of_diagonal = float()
    
    try:
        # 显示当前状态
        status = "等待选择" if not trackObject else "追踪中" if trackObject == 1 else "初始化追踪"
        cv2.putText(display_img, "Status: " + status, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 实时显示框选过程
        if selectObject and ws > 0 and hs > 0:
            cv2.rectangle(display_img, (xs, ys), (xs+ws, ys+hs), (0, 255, 0), 2)
            cv2.putText(display_img, "Selection: {}x{}".format(ws, hs), 
                       (xs, ys-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 执行跟踪
        if trackObject != 0:
            try:
                # 增加HSV掩码范围，提高跟踪稳定性
                mask = cv2.inRange(hsv, np.array((0., 20., 30.)), np.array((180., 255., 255.)))
                
                if trackObject == -1:  # 初次选择目标
                    track_window = (int(xs), int(ys), int(ws), int(hs))
                    maskroi = mask[ys:ys + hs, xs:xs + ws]
                    hsv_roi = hsv[ys:ys + hs, xs:xs + ws]
                    
                    # 使用多个通道计算直方图，提高特征区分度
                    channels = [0, 1]  # 使用H和S通道
                    ranges = [0, 180, 0, 256]  # H和S通道的范围
                    roi_hist = cv2.calcHist([hsv_roi], channels, maskroi, [180, 256], ranges)
                    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
                    trackObject = 1
                    
                    # 显示已选择的区域
                    cv2.rectangle(display_img, (int(xs), int(ys)), 
                                (int(xs+ws), int(ys+hs)), (255, 0, 0), 2)
                    cv2.putText(display_img, "Target Locked", (int(xs), int(ys-10)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                # 执行跟踪
                dst = cv2.calcBackProject([hsv], [0, 1], roi_hist, [0, 180, 0, 256], 1)
                dst &= mask
                
                # 应用均值漂移跟踪
                ret, track_window = cv2.CamShift(dst, track_window, term_crit)
                
                # 更新跟踪结果
                pts = cv2.boxPoints(ret)
                pts = np.int0(pts)
                centerX = float(ret[0][0])
                length_of_diagonal = math.sqrt(float(ret[1][1]) ** 2 + float(ret[1][0]) ** 2)
                
                # 绘制跟踪框和中心点
                cv2.polylines(display_img, [pts], True, (0, 0, 255), 2)
                cv2.circle(display_img, (int(centerX), int(ret[0][1])), 5, (0, 255, 255), -1)
                
                # 显示跟踪信息
                cv2.putText(display_img, "Tracking: ({:.0f}, {:.0f})".format(centerX, ret[0][1]), 
                           (10, display_img.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                rospy.loginfo("追踪状态 - Center: (%.1f, %.1f), Size: %.1f" % 
                             (centerX, ret[0][1], length_of_diagonal))
                
            except Exception as e:
                rospy.logerr("跟踪处理失败: %s", str(e))
                # 跟踪失败时不重置trackObject，保持跟踪状态
                centerX = -1.0
                length_of_diagonal = 0.0
        
        # 显示操作提示
        if not trackObject:
            cv2.putText(display_img, "Click and drag to select target", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('imshow', display_img)
        key = cv2.waitKey(1)
        
        # 添加键盘控制
        if key == ord('r'):  # 按R键重置跟踪
            trackObject = 0
            rospy.loginfo("重置追踪")
        elif key == ord('q'):  # 按Q键退出
            rospy.signal_shutdown("User quit")
            
    except Exception as e:
        rospy.logerr("ExamByCamshift错误: %s", str(e))
        return -1.0, 0.0
        
    return centerX, length_of_diagonal

class image_listenner:
    def __init__(self):
        # 初始化Publisher
        self.twist_pub = None  # 将在初始化完成后创建
        
        # 调整控制参数
        self.threshold = 80
        self.linear_x = 0.3  # 前进速度
        self.angular_z = 0.5  # 转向速度
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
        self.max_track_lost = 30  # 最大丢失帧数
        self.min_target_size = 20  # 最小目标大小
        
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
