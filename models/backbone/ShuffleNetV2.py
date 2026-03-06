# 通道重排，跨group信息交流
import torch
import torch.nn as nn

# 定义通道混洗函数，用于ShuffleNet
def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # 重塑张量形状，将通道分组并重新排列
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()

    # 展平张量
    x = x.view(batchsize, -1, height, width)

    return x

# 定义CBRM模块（Convolution - BatchNormalization - ReLU - MaxPooling）
class CBRM(nn.Module):
    def __init__(self, c1, c2):  # 输入通道数c1，输出通道数c2
        super(CBRM, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False),  # 3x3卷积层
            nn.BatchNorm2d(c2),  # 批归一化层
            nn.ReLU(inplace=True),  # ReLU激活函数
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

    def forward(self, x):
        return self.maxpool(self.conv(x))  # 卷积 - 批归一化 - ReLU - 最大池化

# 定义ShuffleNet中的Shuffle Block模块
class Shuffle_Block(nn.Module):
    def __init__(self, ch_in, ch_out, stride):
        super(Shuffle_Block, self).__init__()

        if not (1 <= stride <= 2):
            raise ValueError('非法的stride值')

        self.stride = stride

        branch_features = ch_out // 2

        assert (self.stride != 1) or (ch_in == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(ch_in, ch_in, kernel_size=3, stride=self.stride, padding=1),  # 深度可分离卷积层
                nn.BatchNorm2d(ch_in),  # 批归一化层
                nn.Conv2d(ch_in, branch_features, kernel_size=1, stride=1, padding=0, bias=False),  # 1x1卷积层
                nn.BatchNorm2d(branch_features),  # 批归一化层
                nn.ReLU(inplace=True),  # ReLU激活函数
            )

        self.branch2 = nn.Sequential(
            nn.Conv2d(ch_in if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),  # 1x1卷积层
            nn.BatchNorm2d(branch_features),  # 批归一化层
            nn.ReLU(inplace=True),  # ReLU激活函数
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),  # 深度可分离卷积层
            nn.BatchNorm2d(branch_features),  # 批归一化层
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),  # 1x1卷积层
            nn.BatchNorm2d(branch_features),  # 批归一化层
            nn.ReLU(inplace=True),  # ReLU激活函数
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)  # 深度可分离卷积

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)  # 按维度1进行分割，将输入分为两个部分
            out = torch.cat((x1, self.branch2(x2)), dim=1)  # 将两个部分连接起来
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)  # 将两个分支的结果连接起来

        out = channel_shuffle(out, 2)  # 执行通道混洗操作

        return out
