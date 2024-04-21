from __future__ import annotations

from typing import Union

import torch
from monai.data import DataLoader as MonaiDataLoader
from torch.utils.data import DataLoader as PyTorchDataLoader

from modules.base.updater import BaseUpdater
from modules.segmentation.module import SegmentationEncoderDecoder, SegmentationModule

DataLoader = Union[MonaiDataLoader, PyTorchDataLoader]


class SegmentationUpdater(BaseUpdater):
    """A simple updater to update parameters in a segmentation module."""

    alias = "SegUpdater"

    def check_module(self, module):
        assert isinstance(module, torch.nn.Module), "The specified module should inherit torch.nn.Module."
        assert isinstance(
            module, (SegmentationModule, SegmentationEncoderDecoder)
        ), "The specified module should inherit SegmentationModule."
        for component in ("criterion", "optimizer"):
            assert getattr(
                module, component, False
            ), f"The specified module should incoporate component/method: {component}"

    def update(self, module, images, masks, modalities=None):
        module.optimizer.zero_grad()
        output = module.forward(images)
        loss = module.criterion(output, masks)
        loss.backward()
        module.optimizer.step()
        return loss.item()
