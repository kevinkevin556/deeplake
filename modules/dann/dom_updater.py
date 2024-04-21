from __future__ import annotations

from typing import Literal

import torch

from modules.base.updater import BaseUpdater
from modules.dann.module import DANNModule


class DomUpdaterDANN(BaseUpdater):
    def __init__(
        self,
        sampling_mode: Literal["sequential", "random_swap", "random_choice"] = "sequential",
    ):
        super().__init__()
        self.sampling_mode = sampling_mode

    @staticmethod
    def grl_lambda(step, max_iter):
        p = float(step) / max_iter
        # grl_lambda = 2.0 / (1.0 + np.exp(-10 * p)) - 1
        grl_lambda = p
        return grl_lambda

    def check_module(self, module):
        assert isinstance(module, torch.nn.Module), "The specified module should inherit torch.nn.Module."
        assert isinstance(module, DANNModule), "The specified module should inherit DANNModule."
        for component in ("criterion", "optimizer", "encoder", "decoder", "dom_classifier", "grl", "adv_loss"):
            assert getattr(
                module, component, False
            ), "The specified module should incoporate component/method: {component}"

    def update(self, module, images, masks, modalities, alpha=1):
        # Set alpha value for the gradient reversal layer and reset gradients
        module.grl.set_alpha(alpha)
        module.optimizer.zero_grad()

        # Segmentation
        source_image, source_mask = images[0], masks[0]
        target_image, _ = images[1], masks[1]
        source_skips, source_feature = module.encoder(source_image)
        _, target_feature = module.encoder(target_image)

        source_output = module.decoder((source_skips, source_feature))
        seg_loss = module.criterion(source_output, source_mask)
        seg_loss.backward(retain_graph=True)

        # Compute adversarial loss for domain classification
        source_domain_pred = module.dom_classifier(module.grl.apply(source_feature))
        target_domain_pred = module.dom_classifier(module.grl.apply(target_feature))
        source_domain_label = torch.zeros(source_domain_pred.shape, device="cuda")
        target_domain_label = torch.ones(target_domain_pred.shape, device="cuda")

        adv_loss = module.adv_loss(source_domain_pred, source_domain_label)
        adv_loss += module.adv_loss(target_domain_pred, target_domain_label)
        adv_loss.backward()

        module.optimizer.step()
        return seg_loss.item(), adv_loss.item()
