#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <nav_msgs/Odometry.h>

class TrackingController {
private:
    ros::NodeHandle nh_;
    ros::Publisher cmd_vel_pub_;
    ros::Subscriber target_odom_sub_;
    ros::Subscriber current_odom_sub_;
    
    nav_msgs::Odometry target_odom_;
    nav_msgs::Odometry current_odom_;
    
    double max_linear_speed_;
    double max_angular_speed_;
    double min_distance_;
    double desired_distance_;
    
    bool target_initialized_;
    bool current_initialized_;

public:
    TrackingController() : nh_("~") {
        // 加载参数
        nh_.param("max_linear_speed", max_linear_speed_, 0.5);
        nh_.param("max_angular_speed", max_angular_speed_, 1.0);
        nh_.param("min_distance", min_distance_, 0.5);
        nh_.param("desired_distance", desired_distance_, 1.0);
        
        // 创建发布者和订阅者
        cmd_vel_pub_ = nh_.advertise<geometry_msgs::Twist>("/cmd_vel", 10);
        target_odom_sub_ = nh_.subscribe("/target/odom", 10, &TrackingController::targetCallback, this);
        current_odom_sub_ = nh_.subscribe("/odom", 10, &TrackingController::currentCallback, this);
    }

    void targetCallback(const nav_msgs::Odometry::ConstPtr& msg) {
        target_odom_ = *msg;
        target_initialized_ = true;
    }

    void currentCallback(const nav_msgs::Odometry::ConstPtr& msg) {
        current_odom_ = *msg;
        current_initialized_ = true;
        if (target_initialized_) {
            track();
        }
    }

    void track() {
        geometry_msgs::Twist cmd_vel;

        // 计算相对位置
        double dx = target_odom_.pose.pose.position.x - current_odom_.pose.pose.position.x;
        double dy = target_odom_.pose.pose.position.y - current_odom_.pose.pose.position.y;
        double distance = sqrt(dx*dx + dy*dy);
        
        // 计算目标角度
        double target_angle = atan2(dy, dx);
        
        // 计算控制命令
        if (distance > min_distance_) {
            // 距离控制
            double linear_speed = max_linear_speed_ * (distance - desired_distance_) / distance;
            cmd_vel.linear.x = std::min(std::max(linear_speed, -max_linear_speed_), max_linear_speed_);
            
            // 方向控制
            cmd_vel.angular.z = max_angular_speed_ * target_angle;
        } else {
            // 停止
            cmd_vel.linear.x = 0;
            cmd_vel.angular.z = 0;
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
