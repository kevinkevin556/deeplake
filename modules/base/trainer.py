from __future__ import annotations

import warnings
from pathlib import Path
from typing import Literal, Union

import numpy as np
import tqdm
from loguru import logger
from monai.data import DataLoader as MonaiDataLoader
from monai.metrics import DiceMetric, Metric
from torch import nn
from torch.utils.data import DataLoader as PyTorchDataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from modules.base.updater import BaseUpdater
from modules.base.validator import BaseValidator

DataLoader = Union[MonaiDataLoader, PyTorchDataLoader]


def get_metric_str(metric, prefix=None, suffix=None):
    metric = metric if isinstance(metric, str) else metric.__class__.__name__
    metric = f"{prefix}/{metric}" if prefix is not None else metric
    metric = f"{metric}:{suffix}" if suffix is not None else metric
    return metric


class TrainLogger:
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
        self.log_file = Path(self.checkpoint_dir) / "train.log"

        self.writer = SummaryWriter(log_dir=checkpoint_dir)
        self.msg = ""

    def catch_msg(self, s):
        self.msg = s[:-1]

    def add_to_summary_writer(self, metric, value, step, prefix=None, suffix=None):
        metric = get_metric_str(metric, prefix, suffix)
        self.writer.add_scalar(metric, value, step)

    def add_to_log_file(self, metric, value, step, prefix=None, suffix=None):
        metric = get_metric_str(metric, prefix, suffix)
        logger.remove()
        logger.add(self.log_file)
        logger.info(f"Step {step} | {metric} = {value}")

    def log_train(self, metric, value, step):
        self.add_to_summary_writer(metric, value, step, prefix="train")

    def log_val(self, metric, suffix, value, step):
        if isinstance(suffix, (list, tuple)):
            logger.remove()
            logger.add(self.log_file)
            logger.info(f"Step {step} | Validation")
            for suf, val in zip(suffix, value):
                self.add_to_summary_writer(metric, val, step, prefix="val", suffix=suf)
                self.add_to_log_file(metric, val, step, suffix=suf)
        else:
            self.add_to_summary_writer(metric, value, step, prefix="val", suffix=suffix)
            self.add_to_log_file(metric, value, step, suffix=suffix)

    def success(self, msg):
        logger.remove()
        logger.add(self.log_file)
        logger.add(self.catch_msg, colorize=True)
        logger.success(msg)
        tqdm.write(self.msg)
        self.msg = ""

    def info(self, msg):
        logger.remove()
        logger.add(self.log_file)
        logger.add(self.catch_msg, colorize=True)
        logger.info(msg)
        tqdm.write(self.msg)
        self.msg = ""


class BaseTrainer:
    def __init__(
        self,
        max_iter: int = 10000,
        eval_step: int = 200,
        metric: Metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False),
        checkpoint_dir: str = "./checkpoints/",
        device: Literal["cuda", "cpu"] = "cuda",
        dev: bool = False,
        validator: BaseValidator | None = None,
    ):
        self.max_iter = max_iter
        self.eval_step = eval_step
        self.metric = metric
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        self.validator = validator if validator else BaseValidator(self.metric, is_train=True, device=self.device)
        if dev:
            self.max_iter = 10
            self.eval_step = 3
            self.checkpoint_dir = "./debug/"
            warnings.warn(
                "Trainer will be executed under developer mode. "
                f"max_iter = {self.max_iter}, "
                f"eval_step = {self.eval_step}, "
                f"checkpoint_dir = {self.checkpoint_dir} ",
                UserWarning,
            )
        self.pbar_description = "Training ({step} / {max_iter} Steps) ({modality_label}) (loss={loss:2.5f})"

    def get_alias(self):
        return getattr(self, "alias", self.__class__.__name__)

    # auxilary function to show training info before training procedure starts
    def show_training_info(self, module, *, train_dataloader, val_dataloader):
        print("--------")
        print("Device:", self.device)  # device is a global variable (not an argument of cli)
        print("# of Training Samples:", len(train_dataloader))
        print("# of Validation Samples:", len(val_dataloader))
        print("Max iteration:", self.max_iter, f"steps (validates per {self.eval_step} steps)")
        print("Checkpoint directory:", self.checkpoint_dir)
        print("Evaluation metric:", self.metric.__class__.__name__)
        module.print_info()
        print("--------")

    def train(
        self,
        module: nn.Module,
        updater: BaseUpdater,
        *,
        train_dataloader: DataLoader | None = None,
        val_dataloader: DataLoader | None = None,
    ):
        self.show_training_info(module, train_dataloader=train_dataloader, val_dataloader=val_dataloader)

        # Initalize progress bar and tensorboard writer
        train_pbar = tqdm(range(self.max_iter), dynamic_ncols=True)
        # writer = SummaryWriter(log_dir=self.checkpoint_dir)
        logger = TrainLogger(self.checkpoint_dir)

        # Initial stage. Note: updater(module) checks the module and returns a partial func of updating parameters.
        best_metric = 0
        module_update = updater(module)

        for step in train_pbar:
            module.train()

            # Backpropagation
            batch = next(iter(train_dataloader))
            images, masks = batch["image"].to(self.device), batch["label"].to(self.device)
            modality_label = batch["modality"][0]
            assert modality_label in {"ct", "mr"}
            modality = 0 if modality_label == "ct" else 1
            loss = module_update(images, masks, modality)

            # Update progress bar and summary writer
            info = {
                "step": step + 1,
                "max_iter": self.max_iter,
                "modality_label": modality_label,
                "loss": loss,
            }
            train_pbar.set_description(self.pbar_description.format(**info))
            logger.log_train(module.criterion, loss, step)

            # Validation
            if ((step + 1) % self.eval_step == 0) or (step == self.max_iter - 1):
                val_metrics = self.validator(module, val_dataloader, global_step=step)
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
