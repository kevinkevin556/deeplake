from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from monai.data import DataLoader
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, Metric
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss
from torch.optim import SGD, Adam, AdamW
from tqdm import tqdm

from lib.discrepancy.mmd import MMD
from modules.base.trainer import BaseTrainer, TrainLogger
from modules.base.updater import BaseUpdater


class MMDModule(nn.Module):
    alias = "MMD"

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
        super().__init__()
        self.roi_size = roi_size
        self.sw_batch_size = sw_batch_size

        self.encoder = encoder
        self.decoder = decoder
        self.ct_criterion = ct_criterion
        self.mr_criterion = mr_criterion
        self.discrepancy = MMD(gamma=None)

        # Set up the optimizer for training
        self.params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.lr = lr
        if optimizer == "AdamW":
            self.optimizer = AdamW(self.params, lr=self.lr)
        elif optimizer == "Adam":
            self.optimizer = Adam(self.params, lr=self.lr)
        elif optimizer == "SGD":
            self.optimizer = SGD(self.params, lr=self.lr)
        else:
            raise ValueError("The specified optimizer is not current supported.")

        if pretrained:
            self.load(pretrained)

        self.to(device)

    # Define the forward pass for the module
    def forward(self, x):
        skip_outputs, feature = self.encoder(x)
        output = self.decoder((skip_outputs, feature))
        return output

    # Inference using the sliding window approach
    def inference(self, x):
        self.eval()
        return sliding_window_inference(x, self.roi_size, self.sw_batch_size, self.forward)

    # Save the state of the model components
    def save(self, checkpoint_dir):
        torch.save(self.encoder.state_dict(), os.path.join(checkpoint_dir, "encoder_state.pth"))
        torch.save(self.decoder.state_dict(), os.path.join(checkpoint_dir, "decoder_state.pth"))

    # Load the state of the model components
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


class MMDUpdater(BaseUpdater):
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
        ct_skip_outputs, ct_feature = module.encoder(ct_image)
        ct_output = module.decoder((ct_skip_outputs, ct_feature))
        mr_skip_outputs, mr_feature = module.encoder(mr_image)
        mr_output = module.decoder((mr_skip_outputs, mr_feature))

        # Compute segmentation losses for CT and MR images
        ct_seg_loss = module.ct_criterion(ct_output, ct_mask)
        mr_seg_loss = module.mr_criterion(mr_output, mr_mask)
        # Total segmentation loss is the sum of individual losses
        seg_loss = ct_seg_loss + mr_seg_loss
        seg_loss.backward(retain_graph=True)

        # Compute Discrepancy
        ct_feature = ct_feature.reshape(-1, ct_feature.size(1) * ct_feature.size(2) * ct_feature.size(3))
        ct_feature = F.normalize(ct_feature)
        mr_feature = mr_feature.reshape(-1, mr_feature.size(1) * mr_feature.size(2) * mr_feature.size(3))
        mr_feature = F.normalize(mr_feature)
        discrepancy = module.discrepancy(torch.Tensor(ct_feature), torch.Tensor(mr_feature))
        discrepancy.backward()

        # Update the model parameters
        module.optimizer.step()
        return seg_loss.item(), discrepancy.item()


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


# Trainer class for MMD model
class MMDTrainer(BaseTrainer):
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
            "(seg_loss={seg_loss:2.5f}, discrepancy={discrepancy:2.5f})"
            # "(discrepancy={discrepancy})"
        )

    # Function to display training information
    def show_training_info(self, module, *, ct_dataloader, mr_dataloader):
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
        ct_dataloader: tuple[DataLoader, DataLoader] | None = None,
        mr_dataloader: tuple[DataLoader, DataLoader] | None = None,
    ):
        # Display training information and initialize metrics
        self.show_training_info(module, ct_dataloader=ct_dataloader, mr_dataloader=mr_dataloader)

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
            seg_loss, discrepancy = module_update(images, masks, modalities)

            # Update training progress description
            info = {
                "step": step,
                "max_iter": self.max_iter,
                "modality1": batch1["modality"][0],
                "modality2": batch2["modality"][0],
                "seg_loss": seg_loss,
                "discrepancy": discrepancy,
            }
            train_pbar.set_description(self.pbar_description.format(**info))
            logger.log_train("seg_loss", seg_loss, step)
            logger.log_train("discrepancy", discrepancy, step)

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
