from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import torch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import SGD, Adam, AdamW

from lib.cyclegan.utils import load_cyclegan
from lib.loss.info_nce import InfoNCE
from lib.loss.target_adaptative_loss import TargetAdaptativeLoss
from lib.misc import Concat


class ContrasiveModule(nn.Module):
    def __init__(
        self,
        net: nn.Module = None,
        roi_size: tuple = (512, 512),
        sw_batch_size: int = 1,
        ct_criterion: _Loss = TargetAdaptativeLoss(num_classes=4, background_classes=[0, 1, 2]),
        mr_criterion: _Loss = TargetAdaptativeLoss(num_classes=4, background_classes=[0, 3]),
        contrast_loss: _Loss = InfoNCE(negative_mode="unpaired"),
        optimizer: str = "AdamW",
        lr: float = 0.0001,
        device: Literal["cuda", "cpu"] = "cuda",
        pretrained: Path | None = None,
    ):
        super().__init__()
        self.roi_size = roi_size
        self.sw_batch_size = sw_batch_size

        net = net.to(device)
        self.encoder = net.encoder  # feature extractor
        self.decoder = (
            Concat(net.decoder, net.segmentation_head) if getattr(net, "segmentation_head") else net.decoder
        )  # predictor

        self.ct_criterion = ct_criterion
        self.mr_criterion = mr_criterion
        self.contrast_loss = contrast_loss

        # Set up the optimizer for training
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.lr = lr
        if optimizer == "AdamW":
            self.optimizer = AdamW(params, lr=self.lr)
        elif optimizer == "Adam":
            self.optimizer = Adam(params, lr=self.lr)
        elif optimizer == "SGD":
            self.optimizer = SGD(params, lr=self.lr, weight_decay=0.0002, momentum=0.9)
        else:
            raise ValueError("The specified optimizer is not current supported.")

        if pretrained:
            self.load(pretrained)

        # placeholders for prototypes
        self.ct_prototypes = {}
        self.ct_cluster_features = {}
        self.mr_prototypes = {}
        self.mr_cluster_features = {}

        # temperatures used when calculating NCE for each cluster
        self.ct_prototypes_t = {}
        self.mr_prototypes_t = {}

    def forward(self, x):
        encoded = self.encoder(x)
        output = self.decoder(*encoded)
        return output

    def train(self, mode=True):
        self.encoder.train(mode)
        self.decoder.train(mode)

    def inference(self, x, modality):
        # Inference using the sliding window approach
        self.eval()
        return sliding_window_inference(x, self.roi_size, self.sw_batch_size, self.forward)

    def save(self, checkpoint_dir):
        torch.save(self.encoder.state_dict(), os.path.join(checkpoint_dir, "encoder_state.pth"))
        torch.save(self.decoder.state_dict(), os.path.join(checkpoint_dir, "decoder_state.pth"))

    def load(self, checkpoint_dir):
        try:
            self.encoder.load_state_dict(torch.load(os.path.join(checkpoint_dir, "encoder_state.pth")))
            self.decoder.load_state_dict(torch.load(os.path.join(checkpoint_dir, "decoder_state.pth")))
        except Exception as e:
            raise e

    # Display information about the encoder, decoder, optimizer, and losses
    def print_info(self):
        print("Module Encoder:", self.encoder.__class__.__name__)
        print("       Decoder:", self.decoder.__class__.__name__)
        print("Optimizer:", self.optimizer.__class__.__name__, f"(lr = {self.lr})")
        print("Segmentation Loss:", {"ct": self.ct_criterion, "mr": self.mr_criterion})
        print("Contrastive Loss:", self.contrast_loss)


class CycleGanContrasiveModule(ContrasiveModule):
    def __init__(
        self,
        cyclegan_checkpoints_dir: str,
        net: nn.Module = None,
        roi_size: tuple = (512, 512),
        sw_batch_size: int = 1,
        ct_criterion: _Loss = TargetAdaptativeLoss(num_classes=4, background_classes=[0, 1, 2]),
        mr_criterion: _Loss = TargetAdaptativeLoss(num_classes=4, background_classes=[0, 3]),
        contrast_loss: _Loss = InfoNCE(negative_mode="unpaired"),
        optimizer: str = "AdamW",
        lr: float = 0.0001,
        device: Literal["cuda", "cpu"] = "cuda",
        pretrained: Path | None = None,
    ):
        super().__init__(
            net, roi_size, sw_batch_size, ct_criterion, mr_criterion, contrast_loss, optimizer, lr, device, pretrained
        )
        self.cyclegan = load_cyclegan(cyclegan_checkpoints_dir, which_epoch="latest")
        self.dice2 = DiceCELoss(softmax=True, to_onehot_y=True)

    def forward(self, x):
        encoded = self.encoder(x)
        output = self.decoder(*encoded)
        return output
