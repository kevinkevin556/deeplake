from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from monai.losses import DiceCELoss
from torch import nn
from torch.nn.modules.loss import _Loss

from lib.cycle_gan.get_options import get_option
from lib.loss.target_adaptative_loss import TargetAdaptativeLoss
from modules.dann.module import DANNModule
from modules.dann.part_updater import PartUpdaterDANN
from networks.cyclegan.cycle_gan_model import CycleGANModel
from networks.unet import BasicUNetDecoder, BasicUNetEncoder

default_dom_classifier = nn.Sequential(
    nn.Conv2d(256, 32, kernel_size=3, padding=1),
    nn.MaxPool2d(kernel_size=2),
    nn.LeakyReLU(0.01),
    nn.Conv2d(32, 1, kernel_size=3, padding=1),
    nn.MaxPool2d(kernel_size=2),
    nn.LeakyReLU(0.01),
    # nn.Flatten(),
    # nn.Linear(4096, 1),
    nn.AdaptiveAvgPool2d(output_size=1),
)


class CycleGanDANNModule(DANNModule):
    def __init__(
        self,
        cycle_gan_ckpt_dir: str,
        encoder: nn.Module = BasicUNetEncoder(spatial_dims=2, in_channels=1, features=(32, 32, 64, 128, 256, 32)),
        decoder: nn.Module = BasicUNetDecoder(spatial_dims=2, out_channels=4, features=(32, 32, 64, 128, 256, 32)),
        dom_classifier: nn.Module = default_dom_classifier,
        roi_size: tuple = (512, 512),
        sw_batch_size: int = 1,
        ct_criterion: _Loss = TargetAdaptativeLoss(num_classes=4, background_classes=[0, 1, 2]),
        mr_criterion: _Loss = TargetAdaptativeLoss(num_classes=4, background_classes=[0, 3]),
        optimizer: str = "AdamW",
        lr: float = 0.0001,
        default_forward_branch: int = 0,
        device: Literal["cuda", "cpu"] = "cuda",
        pretrained: Path | None = None,
    ):
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            dom_classifier=dom_classifier,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            ct_criterion=ct_criterion,
            mr_criterion=mr_criterion,
            optimizer=optimizer,
            lr=lr,
            default_forward_branch=default_forward_branch,
            device=device,
            pretrained=pretrained,
        )
        self.cycle_gan = CycleGANModel()
        options = get_option(Path(cycle_gan_ckpt_dir) / "opt.txt")
        options.isTrain = False
        options.checkpoints_dir = Path(cycle_gan_ckpt_dir).parent.absolute()
        self.cycle_gan.initialize(options)
        self.cycle_gan.load_networks("latest")
        self.dice2 = DiceCELoss(softmax=True, to_onehot_y=True)


class PartUpdaterCycleGanDANN(PartUpdaterDANN):
    def __init__(self):
        super().__init__()
        self.sampling_mode = "sequential"

    @staticmethod
    def grl_lambda(step, max_iter):
        p = float(step) / max_iter
        grl_lambda = 2.0 / (1.0 + np.exp(-8 * p)) - 1
        return grl_lambda

    def check_module(self, module):
        super().check_module(module)
        assert isinstance(module, CycleGanDANNModule), "The specified module should inherit CycleGanDANNModule."
        assert getattr(module, "cycle_gan", False), "The specified module should incoporate component/method: cycle_gan"

    def update(self, module, images, masks, modalities, alpha):

        #####
        # TODO -
        #   The code should be cleaned and clarified with commensts.
        #   There are some obsolute and redundant variables in this function.
        #####

        module.grl.set_alpha(alpha)
        masks = list(masks)
        module.optimizer.zero_grad()

        # decoder branch
        _features = {}
        _seg_losses = {}
        _dom_pred_logits = {}
        _output = {}

        module.cycle_gan.set_input({"A": images[0], "B": images[1], "A_paths": "", "B_paths": ""})
        module.cycle_gan.forward()

        fake_images = [module.cycle_gan.fake_B, module.cycle_gan.fake_A]
        # foreground = {"ct": set(module.ct_criterion.foreground), "mr": set(module.mr_criterion.foreground)}
        # num_classes = module.ct_criterion.num_classes

        for i in [0, 1]:
            m = modalities[i][0]
            skip_outputs, _features[i] = module.encoder(images[i])
            _output[i] = module.decoder((skip_outputs, _features[i]))

            if m == "ct":
                _seg_losses[i] = module.ct_criterion(_output[i], masks[i])
                fake_images[i].require_grad = False
                pseudo_softmax = module.decoder(module.encoder(fake_images[i]))
                masks[i] += torch.argmax(pseudo_softmax, dim=1, keepdim=True) * (masks[i] == 0)
                # pseudo_label = torch.argmax(pseudo_softmax, dim=1, keepdim=True)
                # pseudo_foreground = set(torch.unique(pseudo_label).cpu().numpy()) - {0}
                _seg_losses[i] += module.dice2(_output[i], masks[i])
            else:
                _seg_losses[i] = module.mr_criterion(_output[i], masks[i])
                fake_images[i].require_grad = False
                pseudo_softmax = module.decoder(module.encoder(fake_images[i]))
                masks[i] += torch.argmax(pseudo_softmax, dim=1, keepdim=True) * (masks[i] == 0)
                # pseudo_label = torch.argmax(pseudo_softmax, dim=1, keepdim=True)
                # pseudo_foreground = set(torch.unique(pseudo_label).cpu().numpy()) - {0}
                _seg_losses[i] += module.dice2(_output[i], masks[i])
            _seg_losses[i].backward(retain_graph=True)

        seg_loss = _seg_losses[0] + _seg_losses[1]
        # seg_loss.backward(retain_graph=True)

        # Domain Classifier branch: predict domain equivalent
        for i in [0, 1]:
            _dom_pred_logits[i] = module.dom_classifier(module.grl.apply(_features[i]))
        ct_shape = _dom_pred_logits[0].shape
        mr_shape = _dom_pred_logits[1].shape
        dom_pred_logits = torch.cat([_dom_pred_logits[0], _dom_pred_logits[1]])
        if modalities[0][0] == "ct":
            dom_true_label = torch.cat((torch.zeros(ct_shape, device="cuda"), torch.ones(mr_shape, device="cuda")))
        else:
            dom_true_label = torch.cat((torch.ones(ct_shape, device="cuda"), torch.zeros(mr_shape, device="cuda")))
        adv_loss = module.adv_loss(dom_pred_logits, dom_true_label)
        adv_loss.backward()

        module.optimizer.step()
        return seg_loss.item(), adv_loss.item()
