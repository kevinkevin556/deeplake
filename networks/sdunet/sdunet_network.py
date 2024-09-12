import torch
import torch.nn as nn
import torch.nn.functional as F

from .sdunet_block import SDCN_block as SDCN
from .sdunet_block import conv_block as conv

""" Spatial Deformable Kernel-based U-Net """


class SDUNet(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    Args:
        in_channels (int) : input channals / data channals
        out_classes (int) : output channals / segmentation classes
        hidden_features (int) : basic hidden channals
        keep_prob (float) : DropBlock
        drop_size (int) :
    """

    def __init__(
        self, in_channels: int = 1, out_classes: int = 2, hidden_features: int = 4, keep_prob: float = 0.5, drop_size=3
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_classes = out_classes
        self.hidden_features = hidden_features
        layer_channels = [
            hidden_features,
            2 * hidden_features,
            4 * hidden_features,
            8 * hidden_features,
            16 * hidden_features,
        ]

        self.init_conv = conv(in_channels, layer_channels[0])
        self.down1 = SDCN(layer_channels[0], layer_channels[1], keep_prob, drop_size)
        self.down2 = conv(layer_channels[1], layer_channels[2])
        self.down3 = conv(layer_channels[2], layer_channels[3])
        self.down4 = conv(layer_channels[3], layer_channels[4])

        self.up1 = conv(layer_channels[4] + layer_channels[3], layer_channels[3])
        self.up2 = conv(layer_channels[3] + layer_channels[2], layer_channels[2])
        self.up3 = conv(layer_channels[2] + layer_channels[1], layer_channels[1])
        self.up4 = SDCN(layer_channels[1] + layer_channels[0], layer_channels[0], keep_prob, drop_size)

        self.pooling = nn.MaxPool2d(2)
        self.upsampling = nn.Upsample(scale_factor=2, mode="bilinear")
        self.out_conv = nn.Conv2d(layer_channels[0], out_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.init_conv(x)
        x2 = self.pooling(x1)
        x2 = self.down1(x2)

        x3 = self.pooling(x2)
        x3 = self.down2(x3)

        x4 = self.pooling(x3)
        x4 = self.down3(x4)

        x5 = self.pooling(x4)
        x5 = self.down4(x5)

        x = self.upsampling(x5)
        x = torch.cat((x, x4), dim=1)
        x = self.up1(x)

        x = self.upsampling(x)
        x = torch.cat((x, x3), dim=1)
        x = self.up2(x)

        x = self.upsampling(x)
        x = torch.cat((x, x2), dim=1)
        x = self.up3(x)

        x = self.upsampling(x)
        x = torch.cat((x, x1), dim=1)
        x = self.up4(x)

        out = self.out_conv(x)

        return out
