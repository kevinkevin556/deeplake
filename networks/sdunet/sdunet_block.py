import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .sdunet_sconv import SpatialDeformConv as SDConv

""" SDCN Block """


class conv_block(nn.Module):
    """Standard Convolution Block"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class SDCN_block(nn.Module):
    """parallel connection & pointwise convolution & DropBlock"""

    def __init__(self, in_ch: int = 1, out_ch: int = 2, keep_prob: float = 0.8, drop_size=3):
        super().__init__()
        self.deform_conv = nn.Sequential(
            SDConv(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            SDConv(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        # self.deform_conv = SDConv(in_ch, out_ch, kernel_size = 3, padding = 1),
        # self.bn = nn.BatchNorm2d(out_ch),
        # self.relu = nn.ReLU(inplace = True),
        # self.deform_conv = SDConv(out_ch, out_ch, kernel_size = 3, padding = 1),
        # self.bn = nn.BatchNorm2d(out_ch),
        # self.relu = nn.ReLU(inplace = True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        self.pconv = nn.Conv2d(2 * out_ch, out_ch, kernel_size=1, padding="same", stride=1)
        self.dropblock = DropBlock2D(keep_prob, drop_size)

    def forward(self, x):
        x1 = self.deform_conv(x)
        x2 = self.conv(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.pconv(x)
        x = self.dropblock(x)
        return x


""" DropBlock """
""" Source : https://github.com/miguelvr/dropblock """


class DropBlock2D(nn.Module):
    r"""Randomly zeroes 2D spatial blocks of the input tensor.
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout."""

    def __init__(self, drop_prob, block_size):
        super(DropBlock2D, self).__init__()

        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):

        assert x.dim() == 4, "Expected input with 4 dimensions (bsize, channels, height, width)"

        if not self.training or self.drop_prob == 0.0:
            return x
        else:
            # get gamma value
            gamma = self._compute_gamma(x)

            # sample mask
            mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()

            # place mask on input device
            mask = mask.to(x.device)

            # compute block mask
            block_mask = self._compute_block_mask(mask)

            # apply block mask
            out = x * block_mask[:, None, :, :]

            # scale output
            out = out * block_mask.numel() / block_mask.sum()

            return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(
            input=mask[:, None, :, :],
            kernel_size=(self.block_size, self.block_size),
            stride=(1, 1),
            padding=self.block_size // 2,
        )

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]

        block_mask = 1 - block_mask.squeeze(1)

        return block_mask

    def _compute_gamma(self, x):
        return self.drop_prob / (self.block_size**2)
