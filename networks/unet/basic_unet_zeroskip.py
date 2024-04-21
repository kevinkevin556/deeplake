from __future__ import annotations

from collections.abc import Sequence

import torch
from monai.networks.layers.factories import Conv
from monai.networks.nets.basic_unet import Down, TwoConv, UpCat
from monai.utils import ensure_tuple_rep
from torch import nn

from networks.base import SegmentationModel


class BasicUNetZeroSkip(SegmentationModel):
    """An smp-ized monai BasicUnet"""

    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        features: Sequence[int] = (32, 32, 64, 128, 256, 32),
        act: str | tuple = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: str | tuple = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: float | tuple = 0.0,
        upsample: str = "deconv",
    ):
        super().__init__()
        self.encoder = BasicUNetZeroSkipEncoder(spatial_dims, in_channels, features, act, norm, bias, dropout)
        self.decoder = BasicUNetZeroSkipDecoder(
            spatial_dims, out_channels, features, act, norm, bias, dropout, upsample
        )
        self.segmentation_head = nn.Identity()
        self.classification_head = None


class BasicUNetZeroSkipEncoder(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        features: Sequence[int] = (32, 32, 64, 128, 256, 32),
        act: str | tuple = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: str | tuple = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: float | tuple = 0.0,
    ):
        super().__init__()
        fea = ensure_tuple_rep(features, 6)
        print(f"BasicUNetNoSkip features: {fea}.")
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
        skips = (
            torch.zeros_like(x0),
            torch.zeros_like(x1),
            torch.zeros_like(x2),
            torch.zeros_like(x3),
        )
        enc_hidden = x4
        return skips, enc_hidden


class BasicUNetZeroSkipDecoder(nn.Module):
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
        print(f"BasicUNetNoSkip features: {fea}.")
        self.upcat_4 = UpCat(spatial_dims, fea[4], 0, fea[3], act, norm, bias, dropout, upsample)
        self.upcat_3 = UpCat(spatial_dims, fea[3], 0, fea[2], act, norm, bias, dropout, upsample)
        self.upcat_2 = UpCat(spatial_dims, fea[2], 0, fea[1], act, norm, bias, dropout, upsample)
        self.upcat_1 = UpCat(spatial_dims, fea[1], 0, fea[5], act, norm, bias, dropout, upsample, halves=False)
        self.final_conv = Conv["conv", spatial_dims](fea[5], out_channels, kernel_size=1)

    def forward(self, skips, enc_hidden):
        x4 = enc_hidden
        x0, x1, x2, x3 = skips
        u4 = self.upcat_4(x4, torch.zeros_like(x3))
        u3 = self.upcat_3(u4, torch.zeros_like(x2))
        u2 = self.upcat_2(u3, torch.zeros_like(x1))
        u1 = self.upcat_1(u2, torch.zeros_like(x0))
        logits = self.final_conv(u1)
        return logits
