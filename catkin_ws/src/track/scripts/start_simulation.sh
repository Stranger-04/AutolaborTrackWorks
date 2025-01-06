#!/bin/bash

# 设置工作目录
cd $(rospack find track)

# 设置环境变量
export SVGA_VGPU10=0
export LIBGL_ALWAYS_SOFTWARE=1
export GAZEBO_GPU_RAY=0
export OGRE_RTT_MODE=Copy
export GAZEBO_CMT_THREAD_COUNT=1
export DISPLAY=:0

# 清理可能存在的gazebo进程（添加错误处理）
killall -9 gzserver gzclient rosmaster 2>/dev/null || true

# 等待进程完全终止
sleep 2

# 确保source了工作空间的setup.bash
source ../../devel/setup.bash

# 启动仿真
roslaunch track tracking_simulation.launch
