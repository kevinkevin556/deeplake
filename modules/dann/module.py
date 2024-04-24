from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import torch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from torch import nn
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import _Loss
from torch.optim import SGD, Adam, AdamW

from lib.misc import Concat


# Define a gradient reversal layer for domain adaptation in neural networks
class GradientReversalLayer(torch.autograd.Function):
    alpha = 1

    # Initialize with a scaling factor alpha
    def __init__(self, alpha):
        self.set_alpha(alpha)

    # Set the scaling factor alpha
    def set_alpha(self, alpha):
        GradientReversalLayer.alpha = alpha

    @staticmethod
    def forward(ctx, x):
        # Forward pass just returns the input
        ctx.alpha = GradientReversalLayer.alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        # Backward pass reverses the gradient scaled by alpha
        return -ctx.alpha * grad_output


# Define the DANN module for medical image segmentation
class DANNModule(nn.Module):
    alias = "DANN"

    def __init__(
        self,
        net: nn.Module,
        dom_classifier: nn.Module,
        roi_size: tuple,
        sw_batch_size: int,
        ct_criterion: _Loss = DiceCELoss(to_onehot_y=True, softmax=True),
        mr_criterion: _Loss = DiceCELoss(to_onehot_y=True, softmax=True),
        optimizer: str = "AdamW",
        lr: float = 0.0001,
        default_forward_branch: int = 0,
        device: Literal["cuda", "cpu"] = "cuda",
        pretrained: Path | None = None,
    ):
        super().__init__()
        self.default_forward_branch = default_forward_branch
        self.roi_size = roi_size
        self.sw_batch_size = sw_batch_size

        net = net.to(device)
        self.encoder = net.encoder  # feature extractor
        self.decoder = (
            Concat(net.decoder, net.segmentation_head) if getattr(net, "segmentation_head") else net.decoder
        )  # predictor

        self.dom_classifier = dom_classifier.to(device)
        self.grl = GradientReversalLayer(alpha=1)

        self.ct_criterion = ct_criterion
        self.mr_criterion = mr_criterion
        self.adv_loss = BCEWithLogitsLoss()

        # Set up the optimizer for training
        params = (
            list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.dom_classifier.parameters())
        )
        self.lr = lr
        if optimizer == "AdamW":
            self.optimizer = AdamW(params, lr=self.lr)
        elif optimizer == "Adam":
            self.optimizer = Adam(params, lr=self.lr)
        elif optimizer == "SGD":
            self.optimizer = SGD(params, lr=self.lr)
        else:
            raise ValueError("The specified optimizer is not current supported.")

        if pretrained:
            self.load(pretrained)

    # Define the forward pass for the module
    def forward(self, x):
        encoded = self.encoder(x)
        feature = encoded[-1] if isinstance(encoded, (list, tuple)) else encoded

        # prediction branch
        if self.default_forward_branch == 0:
            output = self.decoder(*encoded)
            return output
        # domain classification branch
        elif self.default_forward_branch == 1:
            dom_pred_logits = self.dom_classifier(self.grl.apply(feature))
            return dom_pred_logits
        else:
            raise ValueError(f"Invalid branch number: {self.default_forward_branch}. Expect 0 or 1.")

    def train(self, mode=True):
        self.encoder.train(mode)
        self.decoder.train(mode)
        self.dom_classifier.train(mode)

    # Inference using the sliding window approach
    def inference(self, x):
        self.eval()
        return sliding_window_inference(x, self.roi_size, self.sw_batch_size, self.forward)

    # Save the state of the model components
    def save(self, checkpoint_dir):
        torch.save(self.encoder.state_dict(), os.path.join(checkpoint_dir, "encoder_state.pth"))
        torch.save(self.decoder.state_dict(), os.path.join(checkpoint_dir, "decoder_state.pth"))
        torch.save(self.dom_classifier.state_dict(), os.path.join(checkpoint_dir, "dom_classifier_state.pth"))

    # Load the state of the model components
    def load(self, checkpoint_dir):
        try:
            self.encoder.load_state_dict(torch.load(os.path.join(checkpoint_dir, "encoder_state.pth")))
            self.decoder.load_state_dict(torch.load(os.path.join(checkpoint_dir, "decoder_state.pth")))
            self.dom_classifier.load_state_dict(torch.load(os.path.join(checkpoint_dir, "dom_classifier_state.pth")))
        except Exception as e:
            raise e

    # Display information about the encoder, decoder, optimizer, and losses
    def print_info(self):
        print("Module Encoder:", self.encoder.__class__.__name__)
        print("       Decoder:", self.decoder.__class__.__name__)
        print("Optimizer:", self.optimizer.__class__.__name__, f"(lr = {self.lr})")
        print("Segmentation Loss:", {"ct": self.ct_criterion, "mr": self.mr_criterion})
        print("Discriminator Loss:", self.adv_loss)
