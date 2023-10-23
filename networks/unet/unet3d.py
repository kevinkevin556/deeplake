import torch
import torch.nn.functional as F
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
    def __init__(self, in_ch=1):
        super().__init__()
        self.encoder0 = nn.Sequential(
            Conv3d_IN_ReLU(in_ch, 32, 3),
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
    def __init__(self, out_ch):
        super().__init__()
        self.decoder0 = nn.Sequential(
            Conv3d_IN_ReLU(64 + 128, 64, 3),
            Conv3d_IN_ReLU(64, 64, 3),
            Conv3d_IN_ReLU(64, out_ch, 3),
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
