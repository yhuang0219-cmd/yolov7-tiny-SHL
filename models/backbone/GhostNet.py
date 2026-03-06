import torch
import torch.nn as nn
from models.common import Conv, GhostConv


# Squeeze-and-Excitation (SE) 模块定义
class SeBlock(nn.Module):
    def __init__(self, in_channel, reduction=4):
        super().__init__()

        # Squeeze：全局平均池化层
        self.Squeeze = nn.AdaptiveAvgPool2d(1)

        # Excitation：两层全连接层（1x1卷积）
        self.Excitation = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // reduction, kernel_size=1),  # 1x1卷积，通道数减少
            nn.ReLU(),
            nn.Conv2d(in_channel // reduction, in_channel, kernel_size=1),  # 1x1卷积，通道数恢复
            nn.Sigmoid()  # Sigmoid激活函数
        )

    def forward(self, x):
        y = self.Squeeze(x)  # 对输入进行全局平均池化
        output = self.Excitation(y)  # 通过Excitation模块计算权重
        return x * (output.expand_as(x))  # 使用权重对输入进行加权求和


# Ghost Bottleneck定义
class G_bneck(nn.Module):
    def __init__(self, c1, c2, midc, k=5, s=1, use_se=False):  # ch_in, ch_mid, ch_out, kernel, stride, use_se
        super().__init__()

        assert s in [1, 2]
        c_ = midc  # 中间通道数

        # 构建卷积层序列
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # 1x1卷积，通道数减少到c_
            Conv(c_, c_, 3, s=2, p=1, g=c_, act=False) if s == 2 else nn.Identity(),  # dw卷积，可选stride=2
            # Squeeze-and-Excite模块，如果use_se为True
            SeBlock(c_) if use_se else nn.Sequential(),
            GhostConv(c_, c2, 1, 1, act=False)  # 1x1卷积，通道数恢复到c2
        )

        # 构建shortcut层，确保输入输出通道数和大小一致
        self.shortcut = nn.Identity() if (c1 == c2 and s == 1) else \
            nn.Sequential(Conv(c1, c1, 3, s=s, p=1, g=c1, act=False),
                          Conv(c1, c2, 1, 1, act=False))  # 避免stride=2时通道数改变的情况

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)  # 输出是卷积结果与shortcut相加的结果
