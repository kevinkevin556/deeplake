from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from torch import ones, zeros

from modules.base.updater import BaseUpdater
from modules.dann.module import DANNModule


class PartUpdaterDANN(BaseUpdater):
    def __init__(
        self,
        sampling_mode: Literal["sequential", "random_swap", "random_choice"] = "sequential",
        pixel_level_adv: bool = False,
    ):
        super().__init__()
        self.sampling_mode = sampling_mode
        self.pixel_level_adv = pixel_level_adv

    @staticmethod
    def grl_lambda(step, max_iter):
        p = float(step) / max_iter
        grl_lambda = 2.0 / (1.0 + np.exp(-10 * p)) - 1
        return grl_lambda

    def check_module(self, module):
        assert isinstance(module, torch.nn.Module), "The specified module should inherit torch.nn.Module."
        assert isinstance(module, DANNModule), "The specified module should inherit DANNModule."
        for component in (
            "ct_criterion",
            "mr_criterion",
            "optimizer",
            "encoder",
            "decoder",
            "dom_classifier",
            "grl",
            "adv_loss",
        ):
            assert getattr(
                module, component, False
            ), f"The specified module should incoporate component/method: {component}"

    def update(self, module, images, masks, modalities, alpha=1):
        # Set alpha value for the gradient reversal layer and reset gradients
        module.grl.set_alpha(alpha)
        module.optimizer.zero_grad()

        # Extract features and make predictions for CT and MR images
        ct_image, ct_mask = images[0], masks[0]
        ct_encoded = module.encoder(ct_image)
        ct_output = module.decoder(*ct_encoded)
        ct_feature = ct_encoded[-1] if isinstance(ct_encoded, (list, tuple)) else ct_encoded

        mr_image, mr_mask = images[1], masks[1]
        mr_encoded = module.encoder(mr_image)
        mr_output = module.decoder(*mr_encoded)
        mr_feature = mr_encoded[-1] if isinstance(mr_encoded, (list, tuple)) else mr_encoded

        # Compute segmentation losses for CT and MR images
        ct_seg_loss = module.ct_criterion(ct_output, ct_mask)
        mr_seg_loss = module.mr_criterion(mr_output, mr_mask)

        # Total segmentation loss is the sum of individual losses
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
        dom_true_label = torch.cat((ones(ct_shape, device="cuda"), zeros(mr_shape, device="cuda")))

        # Calculate adversarial loss and perform backward pass
        adv_loss = module.adv_loss(dom_pred_logits, dom_true_label)
        adv_loss.backward()

        # Update the model parameters
        module.optimizer.step()
        return seg_loss.item(), adv_loss.item()
