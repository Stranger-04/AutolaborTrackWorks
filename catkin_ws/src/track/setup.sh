#!/bin/bash

# 设置Python脚本可执行权限
chmod +x $(rospack find track)/scripts/*.py

# 将此命令添加到工作空间的setup.bash中
echo "source $(rospack find track)/setup.sh" >> ../../devel/setup.bash
