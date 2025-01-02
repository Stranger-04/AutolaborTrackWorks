#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <std_msgs/Bool.h>
#include <termios.h>
#include <stdio.h>

class KeyboardControl {
private:
    ros::NodeHandle nh_;
    ros::Publisher cmd_vel_pub_;
    ros::Publisher control_mode_pub_;
    
    double linear_vel_;
    double angular_vel_;
    bool auto_mode_;

public:
    KeyboardControl() : linear_vel_(0.5), angular_vel_(1.0), auto_mode_(true) {
        cmd_vel_pub_ = nh_.advertise<geometry_msgs::Twist>("/cmd_vel", 1);
        control_mode_pub_ = nh_.advertise<std_msgs::Bool>("/control_mode", 1);
        
        ROS_INFO("Keyboard Control:");
        ROS_INFO("------------------");
        ROS_INFO("w/s : 前进/后退");
        ROS_INFO("a/d : 左转/右转");
        ROS_INFO("空格 : 紧急停止");
        ROS_INFO("m : 切换手动/自动模式");
        ROS_INFO("q : 退出");
    }

    int getch() {
        static struct termios oldt, newt;
        tcgetattr(STDIN_FILENO, &oldt);
        newt = oldt;
        newt.c_lflag &= ~(ICANON | ECHO);
        tcsetattr(STDIN_FILENO, TCSANOW, &newt);
        int c = getchar();
        tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
        return c;
    }

    void publishControlMode() {
        std_msgs::Bool mode_msg;
        mode_msg.data = auto_mode_;
        control_mode_pub_.publish(mode_msg);
    }

    void run() {
        char key;
        while(ros::ok()) {
            key = getch();
            geometry_msgs::Twist twist;

            switch(key) {
                case 'w':
                    twist.linear.x = linear_vel_;
                    break;
                case 's':
                    twist.linear.x = -linear_vel_;
                    break;
                case 'a':
                    twist.angular.z = angular_vel_;
                    break;
                case 'd':
                    twist.angular.z = -angular_vel_;
                    break;
                case ' ':
                    twist.linear.x = 0;
                    twist.angular.z = 0;
                    break;
                case 'm':
                    auto_mode_ = !auto_mode_;
                    ROS_INFO("%s模式", auto_mode_ ? "自动" : "手动");
                    publishControlMode();
                    continue;
                case 'q':
                    ROS_INFO("退出键盘控制");
                    return;
                default:
                    continue;
            }

            if (!auto_mode_) {
                cmd_vel_pub_.publish(twist);
            }
        }
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "keyboard_control");
    KeyboardControl keyboard_control;
    keyboard_control.run();
    return 0;
}
