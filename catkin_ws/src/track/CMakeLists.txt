cmake_minimum_required(VERSION 3.0.2)
project(track)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  geometry_msgs
  nav_msgs
  autolabor_pro1_driver
  gazebo_ros
)

catkin_package(
  CATKIN_DEPENDS roscpp geometry_msgs nav_msgs
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_executable(tracking_node src/tracking_node.cpp)
target_link_libraries(tracking_node ${catkin_LIBRARIES})

# 添加键盘控制节点
add_executable(keyboard_control src/keyboard_control.cpp)
target_link_libraries(keyboard_control ${catkin_LIBRARIES})
