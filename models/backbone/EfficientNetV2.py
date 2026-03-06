# EfficientNetV2


import torch
import torch.nn as nn


# 定义卷积-批归一化-激活函数模块
class stem(nn.Module):
    def __init__(self, c1, c2, kernel_size=3, stride=1, groups=1):
        super().__init__()

        # 计算填充值，确保卷积层输出与输入大小一致
        padding = (kernel_size - 1) // 2

        # 创建卷积层，注意设置bias参数为False，因为后面加了批归一化层
        self.conv = nn.Conv2d(c1, c2, kernel_size, stride, padding=padding, groups=groups, bias=False)

        # 批归一化层
        self.bn = nn.BatchNorm2d(c2, eps=1e-3, momentum=0.1)

        # Swish激活函数
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        # 使用卷积层
        x = self.conv(x)
        # 使用批归一化层
        x = self.bn(x)
        # 使用Swish激活函数
        x = self.act(x)
        return x


# 定义Drop Path操作，用于随机丢弃部分特征
def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x

    keep_prob = 1 - drop_prob

    shape = (x.shape[0],) + (1,) * (x.ndim - 1)

    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)

    random_tensor.floor_()  # binarize

    output = x.div(keep_prob) * random_tensor

    return output


# 定义DropPath层
class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# 定义Squeeze-and-Excite模块
class SqueezeExcite_efficientv2(nn.Module):
    def __init__(self, c1, c2, se_ratio=0.25, act_layer=nn.ReLU):
        super().__init__()

        self.gate_fn = nn.Sigmoid()

        reduced_chs = int(c1 * se_ratio)

        # 全局平均池化层
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 降维卷积层
        self.conv_reduce = nn.Conv2d(c1, reduced_chs, 1, bias=True)

        # ReLU激活函数
        self.act1 = act_layer(inplace=True)

        # 升维卷积层
        self.conv_expand = nn.Conv2d(reduced_chs, c2, 1, bias=True)

    def forward(self, x):
        # 全局平均池化
        x_se = self.avg_pool(x)
        # 降维卷积
        x_se = self.conv_reduce(x_se)
        # ReLU激活
        x_se = self.act1(x_se)
        # 升维卷积
        x_se = self.conv_expand(x_se)
        # Sigmoid激活
        x_se = self.gate_fn(x_se)
        # 将x_se维度扩展为和x一样的维度
        x = x * (x_se.expand_as(x))
        return x


# Fused-MBConv 将MBConv中的depthwise conv3×3和扩展conv1×1替换为单个常规conv3×3
class FusedMBConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, expansion=1, se_ration=0, dropout_rate=0.2, drop_connect_rate=0.2):
        super().__init__()

        # shortcut 指的是到残差结构，expansion是为了先升维，再卷积，再降维，再残差
        self.has_shortcut = (s == 1 and c1 == c2)  # 只要是步长为1并且输入输出特征图大小相等，就是True就可以使用残差结构连接
        self.has_expansion = expansion != 1  # expansion不为1时，输出特征图维度就为expansion * c1，k倍的c1，扩展维度
        expanded_c = c1 * expansion

        if self.has_expansion:
            self.expansion_conv = stem(c1, expanded_c, kernel_size=k, stride=s)
            self.project_conv = stem(expanded_c, c2, kernel_size=1, stride=1)
        else:
            self.project_conv = stem(c1, c2, kernel_size=k, stride=s)

        self.drop_connect_rate = drop_connect_rate
        if self.has_shortcut and drop_connect_rate > 0:
            self.dropout = DropPath(drop_connect_rate)

    def forward(self, x):
        if self.has_expansion:
            result = self.expansion_conv(x)
            result = self.project_conv(result)
        else:
            result = self.project_conv(x)

        if self.has_shortcut:
            if self.drop_connect_rate > 0:
                result = self.dropout(result)
            result += x

        return result


# MBConv模块
class MBConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, expansion=1, se_ration=0, dropout_rate=0.2, drop_connect_rate=0.2):
        super().__init__()

        self.has_shortcut = (s == 1 and c1 == c2)
        expanded_c = c1 * expansion
        self.expansion_conv = stem(c1, expanded_c, kernel_size=1, stride=1)
        self.dw_conv = stem(expanded_c, expanded_c, kernel_size=k, stride=s, groups=expanded_c)
        self.se = SqueezeExcite_efficientv2(expanded_c, expanded_c, se_ration) if se_ration > 0 else nn.Identity()
        self.project_conv = stem(expanded_c, c2, kernel_size=1, stride=1)
        self.drop_connect_rate = drop_connect_rate
        if self.has_shortcut and drop_connect_rate > 0:
            self.dropout = DropPath(drop_connect_rate)

    def forward(self, x):
        # 先用1x1的卷积增加升维
        result = self.expansion_conv(x)
        # 再用一般的卷积特征提取
        result = self.dw_conv(result)
        # 添加SE模块
        result = self.se(result)
        # 再用1x1的卷积降维
        result = self.project_conv(result)
        # 如果使用shortcut连接，则加入dropout操作
        if self.has_shortcut:
            if self.drop_connect_rate > 0:
                result = self.dropout(result)
            # Shortcut是指到残差结构，输入输出的channel大小相等，这样才能相加
            result += x

        return result
