from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from colorful import green, red
from monai.data import DataLoader
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, Metric
from torch import nn, ones, zeros
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import _Loss
from torch.optim import SGD, Adam, AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from modules.base_trainer import BaseTrainer, TrainLogger
from modules.base_updater import BaseUpdater


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
        feat_extractor: nn.Module,
        predictor: nn.Module,
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

        self.grl = GradientReversalLayer(alpha=1)
        self.feat_extractor = feat_extractor
        self.predictor = predictor
        self.dom_classifier = dom_classifier
        self.ct_criterion = ct_criterion
        self.mr_criterion = mr_criterion
        self.adv_loss = BCEWithLogitsLoss()

        # Set up the optimizer for training
        params = (
            list(self.feat_extractor.parameters())
            + list(self.predictor.parameters())
            + list(self.dom_classifier.parameters())
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

        self.to(device)

    # Define the forward pass for the module
    def forward(self, x):
        # Extract features and apply the predictor or domain classifier based on the branch
        skip_outputs, feature = self.feat_extractor(x)
        branch = self.default_forward_branch
        if branch == 0:
            output = self.predictor((skip_outputs, feature))
            return output
        elif branch == 1:
            dom_pred_logits = self.dom_classifier(self.grl.apply(feature))
            return dom_pred_logits
        else:
            raise ValueError(f"Invalid branch number: {branch}. Expect 0 or 1.")

    # Inference using the sliding window approach
    def inference(self, x):
        self.eval()
        return sliding_window_inference(x, self.roi_size, self.sw_batch_size, self.forward)

    # Save the state of the model components
    def save(self, checkpoint_dir):
        torch.save(self.feat_extractor.state_dict(), os.path.join(checkpoint_dir, "feat_extractor_state.pth"))
        torch.save(self.predictor.state_dict(), os.path.join(checkpoint_dir, "predictor_state.pth"))
        torch.save(self.dom_classifier.state_dict(), os.path.join(checkpoint_dir, "dom_classifier_state.pth"))

    # Load the state of the model components
    def load(self, checkpoint_dir):
        try:
            self.feat_extractor.load_state_dict(torch.load(os.path.join(checkpoint_dir, "feat_extractor_state.pth")))
            self.predictor.load_state_dict(torch.load(os.path.join(checkpoint_dir, "predictor_state.pth")))
            self.dom_classifier.load_state_dict(torch.load(os.path.join(checkpoint_dir, "dom_classifier_state.pth")))
        except Exception as e:
            raise e

    # Display information about the encoder, decoder, optimizer, and losses
    def print_info(self):
        print("Module Encoder:", self.feat_extractor.__class__.__name__)
        print("       Decoder:", self.predictor.__class__.__name__)
        print("Optimizer:", self.optimizer.__class__.__name__, f"(lr = {self.lr})")
        print("Segmentation Loss:", {"ct": self.ct_criterion, "mr": self.mr_criterion})
        print("Discriminator Loss:", self.adv_loss)


class DANNUpdater(BaseUpdater):
    def __init__(
        self,
        sampling_mode: Literal["sequential", "random_swap", "random_choice"] = "sequential",
    ):
        super().__init__()
        self.sampling_mode = sampling_mode

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
            "feat_extractor",
            "predictor",
            "dom_classifier",
            "grl",
            "adv_loss",
        ):
            assert getattr(
                module, component, False
            ), "The specified module should incoporate component/method: {component}"

    def update(self, module, images, masks, modalities, alpha=1):
        # Set alpha value for the gradient reversal layer and reset gradients
        module.grl.set_alpha(alpha)
        module.optimizer.zero_grad()

        # Extract features and make predictions for CT and MR images
        ct_image, ct_mask = images[0], masks[0]
        mr_image, mr_mask = images[1], masks[1]
        ct_skip_outputs, ct_feature = module.feat_extractor(ct_image)
        ct_output = module.predictor((ct_skip_outputs, ct_feature))
        mr_skip_outputs, mr_feature = module.feat_extractor(mr_image)
        mr_output = module.predictor((mr_skip_outputs, mr_feature))

        # Compute segmentation losses for CT and MR images
        ct_seg_loss = module.ct_criterion(ct_output, ct_mask)
        mr_seg_loss = module.mr_criterion(mr_output, mr_mask)
        # Total segmentation loss is the sum of individual losses
        seg_loss = ct_seg_loss + mr_seg_loss
        seg_loss.backward(retain_graph=True)

        # Compute adversarial loss for domain classification
        ct_dom_pred_logits = module.dom_classifier(module.grl.apply(ct_feature))
        mr_dom_pred_logits = module.dom_classifier(module.grl.apply(mr_feature))
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


def get_batch(ct_dataloader, mr_dataloader, mode):
    if mode == "sequential":
        batch1 = next(iter(ct_dataloader))
        batch2 = next(iter(mr_dataloader))
    elif mode == "random_swap":
        batch1 = next(iter(ct_dataloader))
        batch2 = next(iter(mr_dataloader))
        if random.random() > 0.5:
            batch1, batch2 = batch2, batch1
    elif mode == "random_choice":
        batch1 = next(iter(ct_dataloader)) if np.random.random(1) < 0.5 else next(iter(mr_dataloader))
        batch2 = next(iter(ct_dataloader)) if np.random.random(1) < 0.5 else next(iter(mr_dataloader))
    else:
        raise ValueError("Invalid mode.")
    return batch1, batch2


# Trainer class for DANN model
class DANNTrainer(BaseTrainer):
    def __init__(
        self,
        max_iter: int = 10000,
        eval_step: int = 100,
        metric: Metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False),
        checkpoint_dir: str = "./checkpoints/",
        device: Literal["cuda", "cpu"] = "cuda",
        dev: bool = False,
    ):
        super().__init__(max_iter, eval_step, metric, checkpoint_dir, device, dev)
        self.pbar_description = (
            "Training ({step} / {max_iter} Steps) ({modality1},{modality2})"
            "(grl_lambda={grl_lambda:2.3f})"
            "(seg_loss={seg_loss:2.5f}, adv_loss={adv_loss:2.5f})"
        )

    # Function to display training information
    def show_training_info(self, module, ct_dataloader, mr_dataloader):
        print("--------")
        print("Device:", self.device)  # Display the computing device
        print("# of Training Samples:", {"ct": len(ct_dataloader[0]), "mr": len(mr_dataloader[0])})
        print("# of Validation Samples:", {"ct": len(ct_dataloader[1]), "mr": len(mr_dataloader[1])})
        print("Max iteration:", self.max_iter, f"steps (validates per {self.eval_step} steps)")
        print("Checkpoint directory:", self.checkpoint_dir)
        print("Evaluation metric:", self.metric.__class__.__name__)
        module.print_info()
        print("--------")

    # Training process for the DANN model
    def train(
        self,
        module,
        updater,
        *,
        ct_dataloader: tuple(DataLoader, DataLoader) | None = None,
        mr_dataloader: tuple(DataLoader, DataLoader) | None = None,
    ):
        # Display training information and initialize metrics
        self.show_training_info(module, ct_dataloader, mr_dataloader)

        # Initalize progress bar and tensorboard writer
        train_pbar = tqdm(range(self.max_iter), dynamic_ncols=True)
        logger = TrainLogger(self.checkpoint_dir)

        # Initial stage. Note: updater(module) checks the module and returns a partial func of updating parameters.
        best_metric = 0
        module_update = updater(module)

        # Main training loop
        for step in train_pbar:
            module.train()

            # Load batches based on sampling mode
            batch1, batch2 = get_batch(ct_dataloader[0], mr_dataloader[0], mode=updater.sampling_mode)
            images = batch1["image"].to(self.device), batch2["image"].to(self.device)
            masks = batch1["label"].to(self.device), batch2["label"].to(self.device)
            modalities = batch1["modality"], batch2["modality"]

            # Adjust gradient reversal layer lambda based on training progress
            grl_lambda = updater.grl_lambda(step, self.max_iter)
            seg_loss, adv_loss = module_update(images, masks, modalities, alpha=grl_lambda)

            # Update training progress description
            info = {
                "step": step,
                "max_iter": self.max_iter,
                "modality1": batch1["modality"][0],
                "modality2": batch2["modality"][0],
                "grl_lambda": grl_lambda,
                "seg_loss": seg_loss,
                "adv_loss": adv_loss,
            }
            train_pbar.set_description(self.pbar_description.format(**info))
            logger.log_train("seg_loss", seg_loss, step)
            logger.log_train("adv_loss", adv_loss, step)

            # Perform validation at specified intervals and save model if performance improves
            if ((step + 1) % self.eval_step == 0) or (step == self.max_iter - 1):
                val_metrics = self.validator(module, (ct_dataloader[1], mr_dataloader[1]), global_step=step)

                # Update summary writer
                logger.log_val(
                    self.metric,
                    suffix=["Average", "CT", "MR"],
                    value=(val_metrics["mean"], val_metrics["ct"], val_metrics["mr"]),
                    step=step,
                )

                # Select validation metric
                if min(val_metrics["ct"], val_metrics["mr"]) is not np.nan:
                    val_metric = min(val_metrics["ct"], val_metrics["mr"])
                else:
                    val_metric = val_metrics["mean"]

                # Update best metric
                if val_metric > best_metric:
                    module.save(self.checkpoint_dir)
                    logger.success(
                        f"Model saved! Validation: (New) {val_metric:2.7f} > (Old) {best_metric:2.7f} "
                        f"(CT) {val_metrics['ct']:2.7f} (MR) {val_metrics['mr']:2.7f}"
                    )
                    best_metric = val_metric
                else:
                    logger.info(
                        f"No improvement. Validation: (New) {val_metric:2.7f} <= (Old) {best_metric:2.7f} "
                        f"(CT) {val_metrics['ct']:2.7f} (MR) {val_metrics['mr']:2.7f}"
                    )
