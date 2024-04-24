from __future__ import annotations

from collections.abc import Sequence

from torch import nn

from networks.base import SegmentationModel
from networks.unet.basic_unet import BasicUNetDecoder, BasicUNetEncoder, BasicUNetIdSkip
from networks.unet.basic_unet_noskip import (
    BasicUNetNoSkip,
    BasicUNetNoSkipDecoder,
    BasicUNetNoSkipEncoder,
)
from networks.unet.basic_unet_zeroskip import (
    BasicUNetZeroSkip,
    BasicUNetZeroSkipDecoder,
    BasicUNetZeroSkipEncoder,
)


class BasicUNet(SegmentationModel):
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
        skip_conn_type: int | None = 1,
    ):
        super().__init__()
        if skip_conn_type is None:
            self.encoder = BasicUNetNoSkipEncoder(spatial_dims, in_channels, features, act, norm, bias, dropout)
            self.decoder = BasicUNetNoSkipDecoder(
                spatial_dims, out_channels, features, act, norm, bias, dropout, upsample
            )
        elif skip_conn_type == 0:
            self.encoder = BasicUNetZeroSkipEncoder(spatial_dims, in_channels, features, act, norm, bias, dropout)
            self.decoder = BasicUNetZeroSkipDecoder(
                spatial_dims, out_channels, features, act, norm, bias, dropout, upsample
            )
        elif skip_conn_type == 1:
            self.encoder = BasicUNetEncoder(spatial_dims, in_channels, features, act, norm, bias, dropout)
            self.decoder = BasicUNetDecoder(spatial_dims, out_channels, features, act, norm, bias, dropout, upsample)
        else:
            raise ValueError(f"skip_connection take a value among {0, 1, None}.")
        self.segmentation_head = None
        self.classification_head = None
