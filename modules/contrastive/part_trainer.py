from __future__ import annotations

import random
from typing import Literal

import numpy as np
from monai.data import DataLoader
from monai.metrics import DiceMetric, Metric
from torch import Tensor
from tqdm import tqdm

from modules.base.trainer import BaseTrainer, TrainLogger
from modules.base.validator import BaseValidator


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


class PartTrainerContrastive(BaseTrainer):
    def __init__(
        self,
        max_iter: int = 10000,
        eval_step: int = 100,
        metric: Metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False),
        checkpoint_dir: str = "./checkpoints/",
        device: Literal["cuda", "cpu"] = "cuda",
        dev: bool = False,
        validator: BaseValidator | None = None,
    ):
        super().__init__(max_iter, eval_step, metric, checkpoint_dir, device, dev, validator)
        self.pbar_description = (
            "Training ({step} / {max_iter} Steps) ({modality1},{modality2})"
            "(seg_loss={seg_loss:2.5f}, nce_loss={nce_loss:2.5f})"
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
            images = Tensor(batch1["image"]).to(self.device), Tensor(batch2["image"]).to(self.device)
            masks = Tensor(batch1["label"]).to(self.device), Tensor(batch2["label"]).to(self.device)
            modalities = batch1["modality"], batch2["modality"]

            # Adjust gradient reversal layer lambda based on training progress
            seg_loss, nce_loss = module_update(images, masks, modalities)

            # Update training progress description
            info = {
                "step": step,
                "max_iter": self.max_iter,
                "modality1": batch1["modality"][0],
                "modality2": batch2["modality"][0],
                "seg_loss": seg_loss,
                "nce_loss": nce_loss,
            }
            train_pbar.set_description(self.pbar_description.format(**info))
            logger.log_train("seg_loss", seg_loss, step)
            logger.log_train("nce_loss", nce_loss, step)

            # Perform validation at specified intervals and save model if performance improves
            if ((step + 1) % self.eval_step == 0) or (step == self.max_iter - 1):
                module.eval()
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
