from __future__ import annotations

from collections.abc import Sequence
from typing import Sequence, Union

import torch
import torch.nn as nn
from monai.networks.layers.factories import Conv
from monai.networks.nets.basic_unet import Down, TwoConv, UpCat
from monai.utils import ensure_tuple_rep
from torch import nn


class Conv3d_IN_ReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, bias=False, padding=1, **kwargs),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)


class UNet3DEncoder(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.encoder0 = nn.Sequential(
            Conv3d_IN_ReLU(in_channels, 32, 3),
            Conv3d_IN_ReLU(32, 64, 3),
        )
        self.encoder1 = nn.Sequential(
            Conv3d_IN_ReLU(64, 64, 3),
            Conv3d_IN_ReLU(64, 128, 3),
        )
        self.encoder2 = nn.Sequential(
            Conv3d_IN_ReLU(128, 128, 3),
            Conv3d_IN_ReLU(128, 256, 3),
        )
        self.encoder3 = nn.Sequential(
            Conv3d_IN_ReLU(256, 256, 3),
            Conv3d_IN_ReLU(256, 512, 3),
        )
        self.max_pool = nn.MaxPool3d(kernel_size=2)

    def forward(self, x):
        enc0 = self.encoder0(x)

        x1 = self.max_pool(enc0)
        enc1 = self.encoder1(x1)

        x2 = self.max_pool(enc1)
        enc2 = self.encoder2(x2)

        x3 = self.max_pool(enc2)
        enc3 = self.encoder3(x3)

        skips = (enc0, enc1, enc2)
        enc_hidden = enc3
        return skips, enc_hidden


class UNet3DDecoder(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.decoder0 = nn.Sequential(
            Conv3d_IN_ReLU(64 + 128, 64, 3),
            Conv3d_IN_ReLU(64, 64, 3),
            Conv3d_IN_ReLU(64, out_channels, 3),
        )
        self.up_conv1 = nn.ConvTranspose3d(128, 128, 2, stride=2)
        self.decoder1 = nn.Sequential(
            Conv3d_IN_ReLU(128 + 256, 128, 3),
            Conv3d_IN_ReLU(128, 128, 3),
        )
        self.up_conv2 = nn.ConvTranspose3d(256, 256, 2, stride=2)
        self.decoder2 = nn.Sequential(
            Conv3d_IN_ReLU(256 + 512, 256, 3),
            Conv3d_IN_ReLU(256, 256, 3),
        )
        self.up_conv3 = nn.ConvTranspose3d(512, 512, 2, stride=2)

    def forward(self, x):
        skips, enc_hidden = x

        y2 = self.up_conv3(enc_hidden)
        y2 = torch.cat([skips[2], y2], dim=1)
        dec2 = self.decoder2(y2)

        y1 = self.up_conv2(dec2)
        y1 = torch.cat([skips[1], y1], dim=1)
        dec1 = self.decoder1(y1)

        y0 = self.up_conv1(dec1)
        y0 = torch.cat([skips[0], y0], dim=1)
        dec0 = self.decoder0(y0)
        return dec0


class BasicUNetEncoder(nn.Module):
    """The encoder part of BasicUNet from monai."""

    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        features: Sequence[int] = (32, 32, 64, 128, 256, 32),
        act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: Union[str, tuple] = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: Union[float, tuple] = 0.0,
    ):
        super().__init__()
        fea = ensure_tuple_rep(features, 6)
        self.conv_0 = TwoConv(spatial_dims, in_channels, features[0], act, norm, bias, dropout)
        self.down_1 = Down(spatial_dims, fea[0], fea[1], act, norm, bias, dropout)
        self.down_2 = Down(spatial_dims, fea[1], fea[2], act, norm, bias, dropout)
        self.down_3 = Down(spatial_dims, fea[2], fea[3], act, norm, bias, dropout)
        self.down_4 = Down(spatial_dims, fea[3], fea[4], act, norm, bias, dropout)

    def forward(self, x: torch.Tensor):
        x0 = self.conv_0(x)
        x1 = self.down_1(x0)
        x2 = self.down_2(x1)
        x3 = self.down_3(x2)
        x4 = self.down_4(x3)
        skips = (x0, x1, x2, x3)
        enc_hidden = x4
        return skips, enc_hidden


class BasicUNetDecoder(nn.Module):
    """The decoder part of BasicUNet from monai."""

    def __init__(
        self,
        spatial_dims: int = 3,
        out_channels: int = 2,
        features: Sequence[int] = (32, 32, 64, 128, 256, 32),
        act: str | tuple = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: str | tuple = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: float | tuple = 0.0,
        upsample: str = "deconv",
    ):
        super().__init__()
        fea = ensure_tuple_rep(features, 6)
        self.upcat_4 = UpCat(spatial_dims, fea[4], fea[3], fea[3], act, norm, bias, dropout, upsample)
        self.upcat_3 = UpCat(spatial_dims, fea[3], fea[2], fea[2], act, norm, bias, dropout, upsample)
        self.upcat_2 = UpCat(spatial_dims, fea[2], fea[1], fea[1], act, norm, bias, dropout, upsample)
        self.upcat_1 = UpCat(spatial_dims, fea[1], fea[0], fea[5], act, norm, bias, dropout, upsample, halves=False)
        self.final_conv = Conv["conv", spatial_dims](fea[5], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        skips, enc_hidden = x
        x4 = enc_hidden
        x0, x1, x2, x3 = skips
        u4 = self.upcat_4(x4, x3)
        u3 = self.upcat_3(u4, x2)
        u2 = self.upcat_2(u3, x1)
        u1 = self.upcat_1(u2, x0)
        logits = self.final_conv(u1)
        return logits
