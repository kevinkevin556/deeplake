from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from monai.losses import DiceCELoss
from torch import nn
from torch.nn.modules.loss import _Loss

from lib.cyclegan.utils import load_cyclegan
from lib.loss.target_adaptative_loss import TargetAdaptativeLoss
from modules.dann.module import DANNModule
from modules.dann.part_updater import PartUpdaterDANN


class CycleGanDANNModule(DANNModule):
    def __init__(
        self,
        cyclegan_checkpoints_dir: str,
        net: nn.Module = None,
        dom_classifier: nn.Module = nn.Sequential(
            nn.Conv2d(256, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.LeakyReLU(0.01),
            nn.AdaptiveAvgPool2d(output_size=1),
        ),
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
            net=net,
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
        self.cyclegan = load_cyclegan(cyclegan_checkpoints_dir, which_epoch="latest")
        self.dice2 = DiceCELoss(softmax=True, to_onehot_y=True)


class PartUpdaterCycleGanDANN(PartUpdaterDANN):
    def __init__(
        self,
        pixel_level_adv: bool = False,
    ):
        super().__init__()
        self.sampling_mode = "sequential"

        # instead of predicting domain label of the whole image
        # predict the domain label of all pixels
        self.pixel_level_adv = pixel_level_adv

    @staticmethod
    def grl_lambda(step, max_iter):
        p = float(step) / max_iter
        # grl_lambda = 2.0 / (1.0 + np.exp(-8 * p)) - 1
        return p

    def check_module(self, module):
        super().check_module(module)
        assert isinstance(module, CycleGanDANNModule), "The specified module should inherit CycleGanDANNModule."
        assert getattr(module, "cyclegan", False), "The specified module should incoporate component/method: cycle_gan"

    def update(self, module, images, masks, modalities, alpha):

        module.grl.set_alpha(alpha)
        masks = list(masks)
        module.optimizer.zero_grad()

        ct, mr = "A", "B"
        fake_mr = module.cyclegan.generate_image(input_image=images[0], from_domain=ct)
        fake_ct = module.cyclegan.generate_image(input_image=images[1], from_domain=mr)

        for i in (0, 1):
            m = modalities[i][0]

            if m == "ct":
                ct_encoded = module.encoder(images[i])
                ct_output = module.decoder(*ct_encoded)
                ct_feature = ct_encoded[-1] if isinstance(ct_encoded, (list, tuple)) else ct_encoded
                ct_seg_loss = module.ct_criterion(ct_output, masks[i])

                fake_mr.require_grad = False
                pseudo_softmax = module.decoder(*module.encoder(fake_mr))
                pseudo_label = masks[i] + torch.argmax(pseudo_softmax, dim=1, keepdim=True) * (masks[i] == 0)
                ct_seg_loss += module.dice2(ct_output, pseudo_label)
            else:
                mr_encoded = module.encoder(images[i])
                mr_output = module.decoder(*mr_encoded)
                mr_feature = mr_encoded[-1] if isinstance(mr_encoded, (list, tuple)) else mr_encoded
                mr_seg_loss = module.mr_criterion(mr_output, masks[i])

                fake_ct.require_grad = False
                pseudo_softmax = module.decoder(*module.encoder(fake_ct))
                pseudo_label = masks[i] + torch.argmax(pseudo_softmax, dim=1, keepdim=True) * (masks[i] == 0)
                mr_seg_loss += module.dice2(mr_output, pseudo_label)

        seg_loss = ct_seg_loss + mr_seg_loss
        seg_loss.backward(retain_graph=True)

        # Compute adversarial loss for domain classification
        if not self.pixel_level_adv:
            ct_dom_pred_logits = module.dom_classifier(module.grl.apply(ct_feature))
            mr_dom_pred_logits = module.dom_classifier(module.grl.apply(mr_feature))
        else:
            ct_dom_pred_logits = module.dom_classifier(None, module.grl.apply(ct_feature))
            mr_dom_pred_logits = module.dom_classifier(None, module.grl.apply(mr_feature))

        # Combine domain predictions and true labels
        ct_shape, mr_shape = ct_dom_pred_logits.shape, mr_dom_pred_logits.shape
        dom_pred_logits = torch.cat([ct_dom_pred_logits, mr_dom_pred_logits])
        dom_true_label = torch.cat((torch.ones(ct_shape, device="cuda"), torch.zeros(mr_shape, device="cuda")))

        # Calculate adversarial loss and perform backward pass
        adv_loss = module.adv_loss(dom_pred_logits, dom_true_label)
        adv_loss.backward()

        module.optimizer.step()
        return seg_loss.item(), adv_loss.item()
