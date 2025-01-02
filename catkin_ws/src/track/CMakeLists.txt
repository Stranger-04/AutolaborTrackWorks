cmake_minimum_required(VERSION 3.0.2)
project(track)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  geometry_msgs
  nav_msgs
  tf2
  tf2_ros
)

catkin_package(
  CATKIN_DEPENDS roscpp geometry_msgs nav_msgs tf2 tf2_ros
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_executable(tracking_node src/tracking_node.cpp)
target_link_libraries(tracking_node ${catkin_LIBRARIES})
