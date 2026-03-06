from typing import Optional

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import checkpoint

from models.common import window_partition, window_reverse


def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    Args:
        x (Tensor): 输入张量。
        drop_prob (float): 丢弃的概率，范围为0到1。
        training (bool): 模型是否处于训练模式。
    Returns:
        Tensor: 经过Drop Path操作后的张量。
    """
    if drop_prob == 0. or not training:
        # 如果丢弃概率为0或模型不处于训练模式，则直接返回输入张量
        return x
    keep_prob = 1 - drop_prob  # 计算保留的概率
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # 确保适用于不同维度的张量，而不仅仅是2D卷积网络
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # 将随机张量二值化，以确定要保留的路径
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    在应用于残差块的主路径中每个样本的Drop Path（随机深度).
    """

    def __init__(self, drop_prob=None):
        """
        初始化DropPath模块。
        Args:
            drop_prob (float): 丢弃的概率，范围为0到1。
        """
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        """
        在输入张量的主路径中应用Drop Path操作。
        Args:
            x (Tensor): 输入张量。
        Returns:
            Tensor: 经过Drop Path操作后的张量。
        """
        return drop_path_f(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # 第一个全连接层，输入特征数为in_features，输出特征数为hidden_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        # 激活函数，默认为GELU
        self.act = act_layer()
        # 第一个Dropout层，用于防止过拟合
        self.drop1 = nn.Dropout(drop)
        # 第二个全连接层，输入特征数为hidden_features，输出特征数为out_features
        self.fc2 = nn.Linear(hidden_features, out_features)
        # 第二个Dropout层，用于防止过拟合
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        # 第一个全连接层的前向传播
        x = self.fc1(x)
        # 应用激活函数
        x = self.act(x)
        # 应用第一个Dropout层
        x = self.drop1(x)
        # 第二个全连接层的前向传播
        x = self.fc2(x)
        # 应用第二个Dropout层
        x = self.drop2(x)
        return x


class WindowAttention(nn.Module):
    r""" 基于窗口的多头自注意力（W-MSA）模块，具有相对位置偏差。
    支持位移和非位移窗口。

    Args:
        dim (int): 输入通道的数量。
        window_size (tuple[int]): 窗口的高度和宽度。
        num_heads (int): 注意力头的数量。
        qkv_bias (bool, optional): 如果为True，则向查询、键、值添加可学习的偏置。默认值：True
        attn_drop (float, optional): 注意力权重的丢弃率。默认值：0.0
        proj_drop (float, optional): 输出的丢弃率。默认值：0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # [Mh, Mw]
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # 定义相对位置偏差的参数表
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # [2*Mh-1 * 2*Mw-1, nH]

        # 为窗口内的每个令牌获取成对的相对位置索引
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # [2, Mh, Mw]
        coords_flatten = torch.flatten(coords, 1)  # [2, Mh*Mw]
        # [2, Mh*Mw, 1] - [2, 1, Mh*Mw]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, Mh*Mw, Mh*Mw]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [Mh*Mw, Mh*Mw, 2]
        relative_coords[:, :, 0] += self.window_size[0] - 1  # 从0开始偏移
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # [Mh*Mw, Mh*Mw]
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: 输入特征，形状为 (num_windows*B, Mh*Mw, C)
            mask: （0/-inf）形状为 (num_windows, Wh*Ww, Wh*Ww) 的掩码，或者为None
        """
        # [batch_size*num_windows, Mh*Mw, total_embed_dim]
        B_, N, C = x.shape
        # qkv(): -> [batch_size*num_windows, Mh*Mw, 3 * total_embed_dim]
        # reshape: -> [batch_size*num_windows, Mh*Mw, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        # [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        q, k, v = qkv.unbind(0)  # 使torchscript满意（不能使用元组作为张量）

        # transpose: -> [batch_size*num_windows, num_heads, embed_dim_per_head, Mh*Mw]
        # @: 相乘 -> [batch_size*num_windows, num_heads, Mh*Mw, Mh*Mw]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # relative_position_bias_table.view: [Mh*Mw*Mh*Mw,nH] -> [Mh*Mw,Mh*Mw,nH]
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [nH, Mh*Mw, Mh*Mw]
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # mask: [nW, Mh*Mw, Mh*Mw]
            nW = mask.shape[0]  # num_windows
            # attn.view: [batch_size, num_windows, num_heads, Mh*Mw, Mh*Mw]
            # mask.unsqueeze: [1, nW, 1, Mh*Mw, Mh*Mw]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # @: 相乘 -> [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        # transpose: -> [batch_size*num_windows, Mh*Mw, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size*num_windows, Mh*Mw, total_embed_dim]
        # x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = (attn.to(v.dtype) @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer 模块。

    Args:
        dim (int): 输入通道的数量。
        num_heads (int): 注意力头的数量。
        window_size (int): 窗口大小。
        shift_size (int): SW-MSA 的位移大小。
        mlp_ratio (float): MLP 隐藏维度与嵌入维度的比例。
        qkv_bias (bool, optional): 如果为True，则向查询、键、值添加可学习的偏置。默认值：True
        drop (float, optional): 丢弃率。默认值：0.0
        attn_drop (float, optional): 注意力丢弃率。默认值：0.0
        drop_path (float, optional): 随机深度率。默认值：0.0
        act_layer (nn.Module, optional): 激活函数层。默认值：nn.GELU
        norm_layer (nn.Module, optional): 归一化层。默认值：nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size 必须在 0 到 window_size 之间"

        # 第一个归一化层
        self.norm1 = norm_layer(dim)
        # 窗口自注意力层
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)

        # 随机深度（Stochastic Depth）层
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # 第二个归一化层
        self.norm2 = norm_layer(dim)

        # MLP 层
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, attn_mask):
        H, W = self.H, self.W
        B, L, C = x.shape
        assert L == H * W, "输入特征的大小不正确"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # 将特征图填充到窗口大小的倍数
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # 循环位移
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            attn_mask = None

        # 划分窗口
        x_windows = window_partition(shifted_x, self.window_size)  # [nW*B, Mh, Mw, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # [nW*B, Mh*Mw, C]

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # [nW*B, Mh*Mw, C]

        # 合并窗口
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)  # [nW*B, Mh, Mw, C]
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # [B, H', W', C]

        # 反向循环位移
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class SwinStage(nn.Module):
    """
    Swin Transformer 模型中一个阶段的基本模块。

    Args:
        dim (int): 输入通道的数量。
        depth (int): 模块中 Swin Transformer 块的数量。
        num_heads (int): 注意力头的数量。
        window_size (int): 本地窗口大小。
        mlp_ratio (float): MLP 隐藏层维度与嵌入层维度的比例。
        qkv_bias (bool, optional): 如果为True，为查询、键和值添加可学习的偏置。默认值：True
        drop (float, optional): 丢弃率。默认值：0.0
        attn_drop (float, optional): 注意力机制中的丢弃率。默认值：0.0
        drop_path (float | tuple[float], optional): 随机深度丢弃率。默认值：0.0
        norm_layer (nn.Module, optional): 归一化层。默认值：nn.LayerNorm
        downsample (nn.Module | None, optional): 在模块末尾的下采样层。默认值：None
        use_checkpoint (bool): 是否使用检查点来节省内存。默认值：False。
    """

    def __init__(self, dim, c2, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False):
        super().__init__()
        assert dim == c2, r"输入/输出通道数应相同"
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.use_checkpoint = use_checkpoint
        self.shift_size = window_size // 2

        # 构建 Swin Transformer 块列表
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

    def create_mask(self, x, H, W):
        # 创建用于 SW-MSA 注意力机制的注意力掩码
        # 确保 Hp 和 Wp 是 window_size 的整数倍
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        # 创建与特征映射具有相同通道排列顺序的图像掩码，以便后续 window_partition
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # [1, Hp, Wp, 1]
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # [nW, Mh, Mw, 1]
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  # [nW, Mh*Mw]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]
        # [nW, Mh*Mw, Mh*Mw]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)  # 重排输入形状
        attn_mask = self.create_mask(x, H, W)  # 创建注意力掩码
        for blk in self.blocks:
            blk.H, blk.W = H, W  # 设置块的高度和宽度属性
            if not torch.jit.is_scripting() and self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)

        x = x.view(B, H, W, C)  # 重新排列输出形状
        x = x.permute(0, 3, 1, 2).contiguous()  # 将通道维度置于正确的位置

        return x  # 返回输出


class PatchEmbed(nn.Module):
    """
    2D 图像到 Patch 嵌入层
    """

    def __init__(self, in_c=3, embed_dim=96, patch_size=4, norm_layer=None):
        super().__init__()
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.in_chans = in_c
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        _, _, H, W = x.shape

        # 填充
        # 如果输入图片的 H 和 W 不是 patch_size 的整数倍，需要进行填充
        pad_input = (H % self.patch_size[0] != 0) or (W % self.patch_size[1] != 0)
        if pad_input:
            # 填充最后的 3 个维度，(W_left, W_right, H_top, H_bottom, C_front, C_back)
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1],
                          0, self.patch_size[0] - H % self.patch_size[0],
                          0, 0))

        # 下采样 patch_size 倍
        x = self.proj(x)
        B, C, H, W = x.shape
        # 展平: [B, C, H, W] -> [B, C, HW]
        # 转置: [B, C, HW] -> [B, HW, C]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        # 重塑形状: [B, HW, C] -> [B, H, W, C]
        # 排列通道维度: [B, H, W, C] -> [B, C, H, W]
        x = x.view(B, H, W, C)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class PatchMerging(nn.Module):
    r""" Patch 合并层。

    Args:
        dim (int): 输入通道的数量。
        norm_layer (nn.Module, optional): 归一化层。默认值：nn.LayerNorm
    """

    def __init__(self, dim, c2, norm_layer=nn.LayerNorm):
        super().__init__()
        assert c2 == (2 * dim), r"输出通道数应为输入通道数的2倍"
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, C, H, W
        """
        B, C, H, W = x.shape
        # assert L == H * W, "input feature has wrong size"
        x = x.permute(0, 2, 3, 1).contiguous()

        # 填充
        # 如果输入 feature map 的 H、W 不是 2 的整数倍，需要进行填充
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            # 填充最后的 3 个维度，从最后的维度开始，向前移动。
            # (C_front, C_back, W_left, W_right, H_top, H_bottom)
            # 注意这里的 Tensor 通道是 [B, H, W, C]，所以会和官方文档有些不同
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]
        x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C]
        x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]
        x = torch.cat([x0, x1, x2, x3], -1)  # [B, H/2, W/2, 4*C]
        x = x.view(B, -1, 4 * C)  # [B, H/2*W/2, 4*C]

        x = self.norm(x)
        x = self.reduction(x)  # [B, H/2*W/2, 2*C]
        x = x.view(B, int(H / 2), int(W / 2), C * 2)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x
