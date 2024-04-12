from __future__ import annotations

import random
from typing import Literal

import numpy as np
from monai.data import DataLoader
from monai.metrics import DiceMetric, Metric
from tqdm import tqdm

from modules.base.trainer import BaseTrainer, TrainLogger
from modules.validator.dom_adapt import DomValidator


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


class DomTrainerDANN(BaseTrainer):
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
            "Training ({step} / {max_iter} Steps) "
            "(grl_lambda={grl_lambda:2.3f}) "
            "(seg_loss={seg_loss:2.5f}, adv_loss={adv_loss:2.5f})"
        )
        self.validator = DomValidator(metric, is_train=True, device=device)

    # Function to display training information
    def show_training_info(self, module, *, ct_dataloader, mr_dataloader):
        print("--------")
        print("- Device:", self.device)
        print("- # of Training Samples:", {"ct": len(ct_dataloader[0]), "mr": len(mr_dataloader[0])})
        print("- # of Validation Samples:", {"ct": len(ct_dataloader[1]), "mr": len(mr_dataloader[1])})
        print("- Max iteration:", self.max_iter, f"steps (validates per {self.eval_step} steps)")
        print("- Checkpoint directory:", self.checkpoint_dir)
        print("- Evaluation metric:", self.metric.__class__.__name__)
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

        # Initalize progress bar, logger and the best result
        train_pbar = tqdm(range(self.max_iter), dynamic_ncols=True)
        logger = TrainLogger(self.checkpoint_dir)
        best_metric = 0
        # Note: updater(module) checks the module and returns a partial func of updating parameters.
        module_update = updater(module)

        # Main training loop
        for step in train_pbar:
            module.train()

            # Load batches based on sampling mode
            # source = batch1 = ct
            # target = batch2 = mr
            batch1, batch2 = get_batch(ct_dataloader[0], mr_dataloader[0], mode="sequential")
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
                "source": batch1["modality"][0],
                "target": batch2["modality"][0],
                "grl_lambda": grl_lambda,
                "seg_loss": seg_loss,
                "adv_loss": adv_loss,
            }
            train_pbar.set_description(self.pbar_description.format(**info))
            logger.log_train("seg_loss", seg_loss, step)
            logger.log_train("adv_loss", adv_loss, step)

            # Perform validation at specified intervals and save model if performance improves
            if ((step + 1) % self.eval_step == 0) or (step == self.max_iter - 1):
                val_metric = self.validator(module, (ct_dataloader[1], mr_dataloader[1]), global_step=step)

                # Update summary writer
                logger.log_val(self.metric, suffix=["Average"], value=[val_metric], step=step)

                # Update best metric
                if val_metric > best_metric:
                    module.save(self.checkpoint_dir)
                    logger.success(f"Model saved! Validation: (New) {val_metric:2.7f} > (Old) {best_metric:2.7f}")
                    best_metric = val_metric
                else:
                    logger.info(f"No improvement. Validation: (New) {val_metric:2.7f} <= (Old) {best_metric:2.7f}")
