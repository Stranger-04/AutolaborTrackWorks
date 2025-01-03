#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/Bool.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <cmath>

class TrackingController {
private:
    ros::NodeHandle nh_;
    ros::Publisher cmd_vel_pub_;
    ros::Subscriber target_odom_sub_;
    ros::Subscriber current_odom_sub_;
    ros::Subscriber control_mode_sub_;  // 新增
    
    nav_msgs::Odometry target_odom_;
    nav_msgs::Odometry current_odom_;
    
    double max_linear_speed_;
    double max_angular_speed_;
    double min_distance_;
    double max_distance_;
    double angle_threshold_;
    
    bool target_initialized_;
    bool current_initialized_;
    bool auto_mode_;  // 新增

public:
    TrackingController() : nh_("~"), target_initialized_(false), current_initialized_(false), auto_mode_(true) {
        // 加载参数
        loadParameters();
        
        // 修改topic名称，使用被控制车辆的odom作为目标
        cmd_vel_pub_ = nh_.advertise<geometry_msgs::Twist>("/track_car/cmd_vel", 10);
        target_odom_sub_ = nh_.subscribe("/target_car/odom", 10, &TrackingController::targetCallback, this);
        current_odom_sub_ = nh_.subscribe("/track_car/odom", 10, &TrackingController::currentCallback, this);
        
        // 添加模式控制订阅者
        control_mode_sub_ = nh_.subscribe("/control_mode", 1, &TrackingController::controlModeCallback, this);
        
        ROS_INFO("Tracking controller initialized - following /target_car/odom");
    }

    void loadParameters() {
        nh_.param("max_linear_speed", max_linear_speed_, 0.8);
        nh_.param("max_angular_speed", max_angular_speed_, 1.5);
        nh_.param("min_distance", min_distance_, 0.5);
        nh_.param("max_distance", max_distance_, 2.0);
        nh_.param("angle_threshold", angle_threshold_, 0.1);
    }

    void targetCallback(const nav_msgs::Odometry::ConstPtr& msg) {
        target_odom_ = *msg;
        target_initialized_ = true;
    }

    void currentCallback(const nav_msgs::Odometry::ConstPtr& msg) {
        current_odom_ = *msg;
        current_initialized_ = true;
        if (target_initialized_ && auto_mode_) {  // 修改这里
            track();
        }
    }

    // 添加模式控制回调
    void controlModeCallback(const std_msgs::Bool::ConstPtr& msg) {
        auto_mode_ = msg->data;
        ROS_INFO("切换至%s模式", auto_mode_ ? "自动" : "手动");
    }

    double normalizeAngle(double angle) {
        while (angle > M_PI) angle -= 2 * M_PI;
        while (angle < -M_PI) angle += 2 * M_PI;
        return angle;
    }

    void track() {
        geometry_msgs::Twist cmd_vel;

        // 计算相对位置
        double dx = target_odom_.pose.pose.position.x - current_odom_.pose.pose.position.x;
        double dy = target_odom_.pose.pose.position.y - current_odom_.pose.pose.position.y;
        double distance = std::sqrt(dx*dx + dy*dy);

        // 获取当前朝向
        tf2::Quaternion q(
            current_odom_.pose.pose.orientation.x,
            current_odom_.pose.pose.orientation.y,
            current_odom_.pose.pose.orientation.z,
            current_odom_.pose.pose.orientation.w);
        double roll, pitch, yaw;
        tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);

        // 计算目标角度和角度差
        double target_angle = std::atan2(dy, dx);
        double angle_diff = normalizeAngle(target_angle - yaw);

        // 根据距离和角度差计算控制命令
        if (std::fabs(angle_diff) > angle_threshold_) {
            // 当角度偏差较大时，优先调整方向
            cmd_vel.angular.z = std::copysign(
                std::min(max_angular_speed_, std::fabs(angle_diff)), angle_diff);
            cmd_vel.linear.x = 0.0;
        } else {
            // 角度基本对准后，调整线速度和角速度
            if (distance > max_distance_) {
                cmd_vel.linear.x = max_linear_speed_;
            } else if (distance < min_distance_) {
                cmd_vel.linear.x = 0.0;
            } else {
                cmd_vel.linear.x = max_linear_speed_ * 
                    (distance - min_distance_) / (max_distance_ - min_distance_);
            }
            cmd_vel.angular.z = angle_diff;
        }

        cmd_vel_pub_.publish(cmd_vel);
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "tracking_node");
    TrackingController controller;
    ros::spin();
    return 0;
}
