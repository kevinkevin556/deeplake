from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from torch import nn
from torch.nn.modules.loss import _Loss

from lib.discrepancy.cdd import CDD
from lib.loss.entropy_loss import EntropyLoss
from modules.base.updater import BaseUpdater
from modules.mmd.mmd import MMDModule


class CDDModule(MMDModule):
    alias = "CDD"

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        roi_size: tuple,
        sw_batch_size: int,
        ct_criterion: _Loss = DiceCELoss(to_onehot_y=True, softmax=True),
        mr_criterion: _Loss = DiceCELoss(to_onehot_y=True, softmax=True),
        optimizer: str = "AdamW",
        lr: float = 0.0001,
        device: Literal["cuda", "cpu"] = "cuda",
        pretrained: Path | None = None,
    ):
        super().__init__(
            encoder,
            decoder,
            roi_size,
            sw_batch_size,
            ct_criterion,
            mr_criterion,
            optimizer,
            lr,
            device,
            pretrained=None,
        )
        self.discrepancy = CDD()
        self.bn = nn.BatchNorm2d(256, affine=True)
        self.ent_loss = EntropyLoss(logits=True)
        if pretrained:
            self.load(pretrained)
        self.to(device)

    # Define the forward pass for the module
    def forward(self, x):
        skip_outputs, feature = self.encoder(x)
        output = self.decoder((skip_outputs, self.bn(feature)))
        return output

    # Inference using the sliding window approach
    def inference(self, x):
        self.eval()
        return sliding_window_inference(x, self.roi_size, self.sw_batch_size, self.forward)


class CDDUpdater(BaseUpdater):
    def __init__(
        self,
        sampling_mode: Literal["sequential", "random_swap", "random_choice"] = "sequential",
    ):
        super().__init__()
        self.sampling_mode = sampling_mode

    def check_module(self, module):
        assert isinstance(module, torch.nn.Module), "The specified module should inherit torch.nn.Module."
        assert isinstance(module, MMDModule), "The specified module should inherit MMDModule."
        for component in ("ct_criterion", "mr_criterion", "optimizer", "encoder", "decoder", "discrepancy"):
            assert getattr(
                module, component, False
            ), "The specified module should incoporate component/method: {component}"

    def update(self, module, images, masks, modalities):
        # Set alpha value for the gradient reversal layer and reset gradients
        module.optimizer.zero_grad()

        # Extract features and make predictions for CT and MR images
        ct_image, ct_mask = images[0], masks[0]
        mr_image, mr_mask = images[1], masks[1]
        _, ct_repr = module.encoder(ct_image)
        _, mr_repr = module.encoder(mr_image)
        no_skip_outputs = (None, None, None, None)

        num_classes = 4 - 1
        batch_size = ct_image.size(0)
        feature_dim = ct_repr.size(1) * ct_repr.size(2) * ct_repr.size(3)
        feat_repr = F.normalize(torch.cat([ct_repr, mr_repr]), dim=(1, 2, 3))

        ct_output = module.decoder((no_skip_outputs, feat_repr[:batch_size]))
        mr_output = module.decoder((no_skip_outputs, feat_repr[batch_size:]))

        # Compute segmentation losses for CT and MR images
        ct_seg_loss = module.ct_criterion(ct_output, ct_mask)
        mr_seg_loss = module.mr_criterion(mr_output, mr_mask)
        # Total segmentation loss is the sum of individual losses
        seg_loss = ct_seg_loss + mr_seg_loss
        seg_loss.backward(retain_graph=True)

        # Compute entropy loss
        # ct_ent_loss = module.ent_loss(ct_output)
        # mr_ent_loss = module.ent_loss(mr_output)
        # ent_loss = ct_ent_loss + mr_ent_loss
        # ent_loss.backward(retain_graph=True)

        # Generate pseudo labels
        ct_plabel = ct_mask + torch.argmax(ct_output, dim=1, keepdim=True) * (ct_mask == 0)
        mr_plabel = mr_mask + torch.argmax(mr_output, dim=1, keepdim=True) * (mr_mask == 0)

        # Generate masked images and class-specific features
        # and then compute Discrepancy
        ct_class_repr = torch.cat([module.encoder(ct_image * (ct_plabel == c + 1))[1] for c in range(num_classes)])
        ct_class_repr = ct_class_repr.reshape(batch_size * num_classes, feature_dim)

        mr_class_repr = torch.cat([module.encoder(mr_image * (mr_plabel == c + 1))[1] for c in range(num_classes)])
        mr_class_repr = mr_class_repr.reshape(batch_size * num_classes, feature_dim)

        class_repr = torch.Tensor(torch.cat([ct_class_repr, mr_class_repr]))
        class_repr = F.normalize(class_repr, dim=1)
        ct_class_repr = class_repr[0 : batch_size * num_classes]
        mr_class_repr = class_repr[batch_size * num_classes :]

        class_repr_label = torch.cat([torch.ones(batch_size) * c for c in range(num_classes)]).byte()
        discrepancy = module.discrepancy(ct_class_repr, mr_class_repr, class_repr_label, class_repr_label, num_classes)
        w_discrepancy = discrepancy * 0.1
        w_discrepancy.backward()

        # Update the model parameters
        module.optimizer.step()
        return seg_loss.item(), discrepancy.item()
