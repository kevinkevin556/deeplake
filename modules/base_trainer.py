from typing import Literal, Optional

import numpy as np
import torch
import tqdm
from monai.data import Dataloader, decollate_batch
from monai.metrics import DiceMetric, Metric
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from lib.utils.validation import get_output_and_mask
from modules.base_validator import BaseValidator


class BaseTrainer:
    def __init__(
        self,
        num_classes: int,
        max_iter: int = 10000,
        metric: Metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False),
        eval_step: int = 200,
        checkpoint_dir: str = "./checkpoints/",
        device: Literal["cuda", "cpu"] = "cuda",
        data_info: dict = None,
        partially_labelled: bool = False,
    ):
        self.max_iter = max_iter
        self.num_classes = num_classes
        self.metric = metric
        self.eval_step = eval_step
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        self.data_info = data_info
        self.partially_labelled = partially_labelled
        self.pbar_description = "Training ({step} / {max_iter} Steps) ({modality_label}) (loss={loss:2.5f})"

    # auxilary function to show training info before training procedure starts
    def show_training_info(self, module, train_dataloader, val_dataloader):
        print("--------")
        print("Device:", self.device)  # device is a global variable (not an argument of cli)
        print("# of Training Samples:", len(train_dataloader))
        print("# of Validation Samples:", len(val_dataloader))
        print("Max iteration:", self.max_iter, f"steps (validates per {self.eval_step} steps)")
        print("Checkpoint directory:", self.checkpoint_dir)
        print("Evaluation metric:", self.metric.__class__.__name__)
        module.print_info()
        print("--------")

    def setup_data(self, train_data, val_data, *args, **kwargs):
        if isinstance(train_data, Dataset):
            self.train_dataloader = Dataloader(train_data, *args, **kwargs)
        else:
            self.train_dataloader = train_data

        if isinstance(val_data, Dataset):
            self.val_dataloader = Dataloader(val_data, *args, **kwargs)
        else:
            self.val_dataloader = val_data

    def setup_module(self, module, *args, **kwargs):
        module.setup(*args, **kwargs)

    def train(self, module, updater, train_data=None, val_data=None, *args, **kwargs):
        # Set up dataloaders and modules
        self.setup_data(train_data, val_data, *args, **kwargs)
        self.setup_module(module, *args, **kwargs)
        self.show_training_info(module, self.train_dataloader, self.val_dataloader)

        # Initalize progress bar and tensorboard writer
        train_pbar = tqdm(range(self.max_iter), dynamic_ncols=True)
        writer = SummaryWriter(log_dir=self.checkpoint_dir)
        writer.add_scalar(f"train/{self.metric.__class__.__name__}", 0, 0)  # validation metric starts from zero

        # Initial stage. Note: updater(module) checks the module and returns a partial func of updating parameters.
        best_metric = 0
        update = updater(module)

        for step in train_pbar:
            module.train()

            # Backpropagation
            batch = next(iter(self.train_dataloader))
            images, masks = batch["image"].to(self.device), batch["label"].to(self.device)
            modality_label = batch["modality"][0]
            modality = None if not self.partially_labelled else 0 if modality_label == "ct" else 1  # ct -> 0, mr -> 1
            loss = update(images, masks, modality)

            # Update progress bar and summary writer
            _info = {"step": step + 1, "max_iter": self.max_iter, "modality_label": modality_label, "loss": loss}
            train_pbar.set_description(self.pbar_description.format(_info))
            writer.add_scalar(f"train/{module.criterion.__class__.__name__}", loss, step)

            # Validation
            if ((step + 1) % self.eval_step == 0) or (step == self.max_iter - 1):
                validator = BaseValidator(self.metric, is_train=True, partially_labelled=self.partially_labelled)
                val_metrics = validator(module, self.val_dataloader, global_step=step)

                # Update summary writer
                writer.add_scalar(f"val/{self.metric.__class__.__name__}:Average", val_metrics["mean"], step)
                writer.add_scalar(f"val/{self.metric.__class__.__name__}:CT", val_metrics["ct"], step)
                writer.add_scalar(f"val/{self.metric.__class__.__name__}:MR", val_metrics["mr"], step)

                # Select validation metric
                if min(val_metrics["ct"], val_metrics["mr"]) is not np.nan:
                    val_metric = np.min(val_metrics["ct"], val_metrics["mr"])
                else:
                    val_metric = val_metrics["mean"]

                # Update best metric
                if val_metric > best_metric:
                    module.save(self.checkpoint_dir)
                    msg = f"\033[32mModel saved! Validation: (New) {val_metric:2.7f} > (Old) {best_metric:2.7f}\033[0m"
                    best_metric = val_metric
                else:
                    msg = f"\033[31mNo improvement. Validation: (New) {val_metric:2.7f} <= (Old) {best_metric:2.7f}\033[0m"
                msg += f" (CT) {val_metrics['ct']:2.7f} (MR) {val_metrics['mr']:2.7f}"
                tqdm.write(msg)
