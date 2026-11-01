"""使用深度可分离卷积的 U-Net 组件。

接口与标准 U-Net 组件一致，但双卷积由 depthwise + pointwise 实现，从而减少
参数量和计算量。基础 U-Net 结构参考 milesial/Pytorch-UNet；本地实现不改变
服务器 V1.1_122 网络拓扑及权重键。local.1 仅完善结构与形状注释。
"""
# 基础 U-Net 参考：https://github.com/milesial/Pytorch-UNet
import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import DepthwiseSeparableConv


class DoubleConvDS(nn.Module):
    """连续两次执行 ``深度可分离卷积 → BatchNorm → ReLU``。"""

    def __init__(self, in_channels, out_channels, mid_channels=None, kernels_per_layer=1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            DepthwiseSeparableConv(in_channels, mid_channels, kernel_size=3, kernels_per_layer=kernels_per_layer, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=False),
            DepthwiseSeparableConv(mid_channels, out_channels, kernel_size=3, kernels_per_layer=kernels_per_layer, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        return self.double_conv(x)


class DownDS(nn.Module):
    """最大池化下采样后执行两次深度可分离卷积。"""

    def __init__(self, in_channels, out_channels, kernels_per_layer=1):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvDS(in_channels, out_channels, kernels_per_layer=kernels_per_layer)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpDS(nn.Module):
    """上采样、尺寸对齐、拼接编码特征，再执行深度可分离双卷积。"""

    def __init__(self, in_channels, out_channels, bilinear=True, kernels_per_layer=1):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConvDS(in_channels, out_channels, in_channels // 2, kernels_per_layer=kernels_per_layer)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConvDS(in_channels, out_channels, kernels_per_layer=kernels_per_layer)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """1×1 卷积输出头；保持空间尺寸，只改变通道数。"""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
