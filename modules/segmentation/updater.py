from __future__ import annotations

from typing import Union

import torch
from monai.data import DataLoader as MonaiDataLoader
from torch import nn
from torch.utils.data import DataLoader as PyTorchDataLoader

from modules.base.updater import BaseUpdater
from modules.segmentation.module import (
    CycleGanSegmentationModule,
    SegmentationEncoderDecoder,
    SegmentationModule,
)

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


class CycleGanSegmentationUpdater(SegmentationUpdater):
    def __init__(self):
        super().__init__()
        self.sampling_mode = "sequential"

    def check_module(self, module):
        assert isinstance(module, nn.Module), "The specified module should inherit torch.nn.Module."
        assert isinstance(
            module, CycleGanSegmentationModule
        ), "The specified module should inherit CycleGanSegmentationModule."
        for component in ("ct_criterion", "mr_criterion", "optimizer", "cyclegan"):
            assert getattr(
                module, component, False
            ), f"The specified module should incoporate component/method: {component}"

    def update(self, module, images, masks, modalities):
        module.optimizer.zero_grad()

        ct, mr = "A", "B"

        if modalities == 0:
            # Train network with fake MR scans (generated from CT)
            fake_mr = module.cyclegan.generate_image(input_image=images, from_domain=ct)
            ct_images, ct_mask = fake_mr, masks
            ct_output = module.net(ct_images)
            seg_loss = module.ct_criterion(ct_output, ct_mask)
            seg_loss += module.ct_criterion(images, ct_mask)
        else:
            # Train network with real MR scans
            fake_ct = module.cyclegan.generate_image(input_image=images, from_domain=mr)
            mr_images, mr_mask = fake_ct, masks
            mr_output = module.net(mr_images)
            seg_loss = module.mr_criterion(mr_output, mr_mask)
            seg_loss += module.mr_criterion(images, mr_mask)

        # Back-prop
        seg_loss.backward()
        module.optimizer.step()
        return seg_loss.item()
