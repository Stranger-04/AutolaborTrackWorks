#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math

class SimpleLaneChange:
    def __init__(self):
        rospy.init_node('simple_lane_change', anonymous=False)
        
        # 运动参数
        self.linear_speed = 0.3
        self.angular_speed = 0.3
        self.distance_threshold = 0.05  # 位置误差阈值
        
        # 状态机
        self.STATE_FORWARD1 = 0    # 第一段直行
        self.STATE_CHANGING = 1    # 变道
        self.STATE_FORWARD2 = 2    # 第二段直行
        self.STATE_FINISHED = 3    # 完成
        self.current_state = self.STATE_FORWARD1
        
        # 位置跟踪
        self.start_x = 0.0
        self.start_y = 0.0
        self.current_x = 0.0
        self.current_y = 0.0
        self.got_first_position = False
        
        # ROS接口
        self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        
        rospy.loginfo("简单变道节点已初始化")

    def odom_callback(self, data):
        """处理里程计数据"""
        self.current_x = data.pose.pose.position.x
        self.current_y = data.pose.pose.position.y
        
        # 记录起始位置
        if not self.got_first_position:
            self.start_x = self.current_x
            self.start_y = self.current_y
            self.got_first_position = True
        
        self.control_loop()

    def get_distance(self, x1, y1, x2, y2):
        """计算两点间距离"""
        return math.sqrt((x2-x1)**2 + (y2-y1)**2)

    def control_loop(self):
        """控制循环"""
        if not self.got_first_position:
            return

        cmd = Twist()
        
        if self.current_state == self.STATE_FORWARD1:
            # 第一段直行1米
            forward_distance = self.get_distance(self.start_x, self.start_y, 
                                              self.current_x, self.current_y)
            if forward_distance < 1.0:
                cmd.linear.x = self.linear_speed
                rospy.loginfo("第一段直行中: %.2f米", forward_distance)
            else:
                self.current_state = self.STATE_CHANGING
                rospy.loginfo("开始变道")
                
        elif self.current_state == self.STATE_CHANGING:
            # 变道1米
            lateral_distance = abs(self.current_y - self.start_y)
            if lateral_distance < 1.0:
                cmd.linear.x = self.linear_speed * 0.5  # 变道时降低速度
                cmd.angular.z = self.angular_speed
                rospy.loginfo("变道中: %.2f米", lateral_distance)
            else:
                self.current_state = self.STATE_FORWARD2
                rospy.loginfo("开始第二段直行")
                
        elif self.current_state == self.STATE_FORWARD2:
            # 第二段直行1米
            forward_distance = self.get_distance(self.current_x, self.current_y,
                                              self.current_x + 1.0, self.current_y)
            if forward_distance < 1.0:
                cmd.linear.x = self.linear_speed
                rospy.loginfo("第二段直行中: %.2f米", forward_distance)
            else:
                self.current_state = self.STATE_FINISHED
                rospy.loginfo("任务完成")
                
        elif self.current_state == self.STATE_FINISHED:
            # 停止运动
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            
        self.cmd_vel_pub.publish(cmd)

    def run(self):
        """运行节点"""
        rate = rospy.Rate(10)  # 10Hz
        while not rospy.is_shutdown():
            rate.sleep()

if __name__ == '__main__':
    try:
        node = SimpleLaneChange()
        node.run()
    except rospy.ROSInterruptException:
        pass
