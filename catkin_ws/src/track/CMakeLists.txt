cmake_minimum_required(VERSION 3.0.2)
project(track)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  geometry_msgs
  nav_msgs
  gazebo_ros
)

catkin_package(
  CATKIN_DEPENDS roscpp geometry_msgs nav_msgs gazebo_ros
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_executable(tracking_node src/tracking_node.cpp)
add_executable(keyboard_control src/keyboard_control.cpp)

target_link_libraries(tracking_node
  ${catkin_LIBRARIES}
)

target_link_libraries(keyboard_control
  ${catkin_LIBRARIES}
)

catkin_install_python(PROGRAMS
  scripts/vision_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

add_custom_target(make_scripts_executable ALL
  COMMAND chmod +x ${CMAKE_CURRENT_SOURCE_DIR}/scripts/vision_node.py
  COMMENT "Making scripts executable"
)
