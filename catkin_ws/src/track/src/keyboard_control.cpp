#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <termios.h>
#include <stdio.h>
#include <algorithm>

class KeyboardControl {
private:
    ros::NodeHandle nh_;
    ros::Publisher cmd_vel_pub_;
    
    double linear_speed_ = 0.5;
    double angular_speed_ = 1.0;
    const double MAX_LINEAR_SPEED = 1.0;
    const double MAX_ANGULAR_SPEED = 2.0;

public:
    KeyboardControl() {
        cmd_vel_pub_ = nh_.advertise<geometry_msgs::Twist>("cmd_vel", 10);
        printInstruction();
    }

    void printInstruction() {
        ROS_INFO("Keyboard Control:");
        ROS_INFO("------------------");
        ROS_INFO("w : Forward");
        ROS_INFO("s : Backward");
        ROS_INFO("a : Turn Left");
        ROS_INFO("d : Turn Right");
        ROS_INFO("space : Stop");
        ROS_INFO("q : Quit");
    }

    char getKey() {
        struct termios old = {0};
        tcgetattr(0, &old);
        struct termios newt = old;
        newt.c_lflag &= ~(ICANON | ECHO);
        tcsetattr(0, TCSANOW, &newt);
        char c = getchar();
        tcsetattr(0, TCSANOW, &old);
        return c;
    }

    void run() {
        char key;
        geometry_msgs::Twist twist;

        while (ros::ok()) {
            key = getKey();
            twist.linear.x = 0;
            twist.angular.z = 0;

            switch(key) {
                case 'w':
                    twist.linear.x = std::min(linear_speed_, MAX_LINEAR_SPEED);
                    break;
                case 's':
                    twist.linear.x = -std::min(linear_speed_, MAX_LINEAR_SPEED);
                    break;
                case 'a':
                    twist.angular.z = std::min(angular_speed_, MAX_ANGULAR_SPEED);
                    break;
                case 'd':
                    twist.angular.z = -std::min(angular_speed_, MAX_ANGULAR_SPEED);
                    break;
                case ' ':
                    twist.linear.x = 0;
                    twist.angular.z = 0;
                    break;
                case 'q':
                    ROS_INFO("Quit");
                    return;
            }
            cmd_vel_pub_.publish(twist);
            ros::spinOnce();
        }
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "keyboard_control");
    ros::NodeHandle nh;
    ros::Publisher cmd_vel_pub = nh.advertise<geometry_msgs::Twist>("cmd_vel", 10);
    KeyboardControl keyboard_control;
    keyboard_control.run();
    return 0;
}
