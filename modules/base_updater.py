from contextlib import nullcontext
from functools import partial
from typing import Optional

import torch
from torch import TensorType, nn

from modules.segmentation import SegmentationEncoderDecoder, SegmentationModule


class BaseUpdater:
    """Base class of updaters."""

    def __init__(self):
        pass

    def __call__(self, module):
        self.check_module(module)
        return partial(self.update, module=module)

    def check_module(self, module: nn.Module) -> None:
        raise NotImplementedError

    def update(self, module: nn.Module, images: TensorType, masks: TensorType, modalities=Optional[int]) -> float:
        raise NotImplementedError
