"""SmaAt-UNet 云参数反演主网络。

网络由深度可分离卷积 U-Net 与 CBAM 通道/空间注意力组成。编码端保存四级注意力
特征供解码端跳跃连接，输出经 ReLU 限制为非负值。输入和输出均采用 PyTorch
``[batch, channel, height, width]`` 布局；实际通道数由调用方按产品模型配置。

版本历史：模型结构及权重兼容性保持服务器 V1.1_122 不变；
V1.1_122+local.1 仅补充网络拓扑、张量形状与包装器用途注释。
"""

from torch import nn
from unet_parts import OutConv
from unet_parts_depthwise_separable import DoubleConvDS, UpDS, DownDS
from layer import CBAM


class SmaAt_UNet(nn.Module):
    """带 CBAM 注意力的轻量 U-Net。

    Parameters
    ----------
    n_channels : int
        输入特征通道数。
    n_classes : int
        输出通道数；CLP 可为多通道，连续云参数通常为单通道。
    kernels_per_layer : int
        深度卷积的通道扩展倍率。
    bilinear : bool
        True 使用双线性上采样，False 使用转置卷积。
    reduction_ratio : int
        CBAM 通道注意力 MLP 的压缩比例。
    """
    def __init__(self, n_channels, n_classes, kernels_per_layer=2, bilinear=True, reduction_ratio=16):
        super(SmaAt_UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        kernels_per_layer = kernels_per_layer
        self.bilinear = bilinear
        reduction_ratio = reduction_ratio

        self.inc = DoubleConvDS(self.n_channels, 64, kernels_per_layer=kernels_per_layer)
        self.cbam1 = CBAM(64, reduction_ratio=reduction_ratio)
        self.down1 = DownDS(64, 128, kernels_per_layer=kernels_per_layer)
        self.cbam2 = CBAM(128, reduction_ratio=reduction_ratio)
        self.down2 = DownDS(128, 256, kernels_per_layer=kernels_per_layer)
        self.cbam3 = CBAM(256, reduction_ratio=reduction_ratio)
        self.down3 = DownDS(256, 512, kernels_per_layer=kernels_per_layer)
        self.cbam4 = CBAM(512, reduction_ratio=reduction_ratio)
        factor = 2 if self.bilinear else 1
        self.down4 = DownDS(512, 1024 // factor, kernels_per_layer=kernels_per_layer)
        self.cbam5 = CBAM(1024 // factor, reduction_ratio=reduction_ratio)
        self.up1 = UpDS(1024, 512 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up2 = UpDS(512, 256 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up3 = UpDS(256, 128 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up4 = UpDS(128, 64, self.bilinear, kernels_per_layer=kernels_per_layer)

        self.outc = OutConv(64, self.n_classes)
        self.relu = nn.ReLU(inplace = False)

    def forward(self, x):
        """完成编码、注意力跳跃连接和解码，返回非负预测张量。"""
        x1 = self.inc(x)
        x1Att = self.cbam1(x1)
        x2 = self.down1(x1)
        x2Att = self.cbam2(x2)
        x3 = self.down2(x2)
        x3Att = self.cbam3(x3)
        x4 = self.down3(x3)
        x4Att = self.cbam4(x4)
        x5 = self.down4(x4)
        x5Att = self.cbam5(x5)
        x = self.up1(x5Att, x4Att)
        x = self.up2(x, x3Att)
        x = self.up3(x, x2Att)
        x = self.up4(x, x1Att)
        logits = self.outc(x)
        logits = self.relu(logits)
        return logits
    
class ModelWrapper(nn.Module):
    """将空间预测展平为 ``[batch, num_outputs]`` 的兼容包装器。

    该类保留用于历史导出/评估流程，不参与主反演路径；标量输出会先补 batch
    维，避免 ``view`` 丢失样本维度。
    """
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        """调用被包装模型并保留 batch 维后展平其余维度。"""
        output = self.model(x)
        # print("Output shape:", output.shape)  # 打印输出的形状
        if output.dim() == 0:  # 检查输出是否是标量
            output = output.unsqueeze(0)  # 如果是标量，则添加一个维度
        return output.view(output.size(0), -1)  # 展平输出以 (batch_size, num_outputs) 的形式
