#!/bin/bash

# 设置环境变量
export SVGA_VGPU10=0
export LIBGL_ALWAYS_SOFTWARE=1
export GAZEBO_GPU_RAY=0
export OGRE_RTT_MODE=Copy
export GAZEBO_CMT_THREAD_COUNT=1
export DISPLAY=:0

# 清理可能存在的gazebo进程
killall -9 gzserver gzclient rosmaster

# 等待进程完全终止
sleep 2

# 启动仿真
roslaunch track tracking_simulation.launch
