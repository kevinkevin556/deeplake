from __future__ import annotations

from pathlib import Path
from typing import Literal

from torch import nn
from torch.nn.modules.loss import _Loss

from lib.cycle_gan.get_options import get_option
from lib.loss.target_adaptative_loss import TargetAdaptativeLoss
from networks.cyclegan.cycle_gan_model import CycleGANModel
from networks.unet import BasicUNet

from .module import SegmentationModule
from .updater import SegmentationUpdater


class CycleGanSegmentationModule(SegmentationModule):
    def __init__(
        self,
        cycle_gan_ckpt_dir: str,
        net: nn.Module = BasicUNet(spatial_dims=2, in_channels=1, out_channels=4, features=(32, 32, 64, 128, 256, 32)),
        roi_size: tuple = (512, 512),
        sw_batch_size: int = 1,
        ct_criterion: _Loss = TargetAdaptativeLoss(num_classes=4, background_classes=[0, 1, 2]),
        mr_criterion: _Loss = TargetAdaptativeLoss(num_classes=4, background_classes=[0, 3]),
        optimizer: str = "AdamW",
        lr: float = 0.0001,
        device: Literal["cuda", "cpu"] = "cuda",
        pretrained: Path | None = None,
    ):
        super().__init__(
            net=net,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            criterion=None,
            optimizer=optimizer,
            lr=lr,
            device=device,
            pretrained=pretrained,
        )
        self.ct_criterion = ct_criterion
        self.mr_criterion = mr_criterion
        self.cycle_gan = CycleGANModel()

        options = get_option(Path(cycle_gan_ckpt_dir) / "opt.txt")
        options.isTrain = False
        options.checkpoints_dir = Path(cycle_gan_ckpt_dir).parent.absolute()
        self.cycle_gan.initialize(options)
        self.cycle_gan.load_networks("latest")

    def print_info(self):
        print("Module:", self.net.__class__.__name__)
        print("Optimizer:", self.optimizer.__class__.__name__, f"(lr = {self.lr})")
        print("Loss function - CT:", repr(self.ct_criterion))
        print("Loss function - MR:", repr(self.mr_criterion))


class CycleGanSegmentationUpdater(SegmentationUpdater):
    def __init__(self):
        super().__init__()
        self.sampling_mode = "sequential"

    def check_module(self, module):
        assert isinstance(module, nn.Module), "The specified module should inherit torch.nn.Module."
        assert isinstance(
            module, CycleGanSegmentationModule
        ), "The specified module should inherit CycleGanSegmentationModule."
        for component in ("ct_criterion", "mr_criterion", "optimizer", "cycle_gan"):
            assert getattr(
                module, component, False
            ), f"The specified module should incoporate component/method: {component}"

    def update(self, module, images, masks, modalities):
        module.optimizer.zero_grad()

        if modalities == 0:
            # Generate fake images using pretrained CycleGAN
            module.cycle_gan.set_input({"A": images, "B": images, "A_paths": "", "B_paths": ""})
            module.cycle_gan.forward()
            # fake_images = [module.cycle_gan.fake_B, module.cycle_gan.fake_A]

            # Train network with fake MR scans (generated from CT)
            ct_images, ct_mask = images, masks
            ct_output = module.net(ct_images)
            seg_loss = module.ct_criterion(ct_output, ct_mask)
        else:
            # Train network with real MR scans
            mr_images, mr_mask = images, masks
            mr_output = module.net(mr_images)
            seg_loss = module.mr_criterion(mr_output, mr_mask)

        # Back-prop
        seg_loss.backward()
        module.optimizer.step()
        return seg_loss
