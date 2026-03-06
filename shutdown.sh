#!/bin/bash

# 运行YoloV7的命令
python train.py

# 检查上一个命令的退出状态
if [ $? -eq 0 ]; then
    # 如果成功执行，则关机
    echo "Training completed. Shutting down the system in 2 minutes."
    sleep 120
    shutdown now
else
    # 如果执行失败，输出错误信息
    echo "YoloV7 运行失败，不会关机。"
    # 可以根据需要添加更多的错误处理逻辑
fi
