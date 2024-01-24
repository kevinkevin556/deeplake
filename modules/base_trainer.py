import warnings
from pathlib import Path
from typing import Literal, Optional, Tuple, Union

import numpy as np
import tqdm
from monai.data import DataLoader as MonaiDataLoader
from monai.metrics import DiceMetric, Metric
from torch import nn
from torch.utils.data import DataLoader as PyTorchDataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from modules.base_updater import BaseUpdater
from modules.base_validator import BaseValidator

DataLoader = Union[MonaiDataLoader, PyTorchDataLoader]


class BaseTrainer:
    def __init__(
        self,
        max_iter: int = 10000,
        eval_step: int = 200,
        metric: Metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False),
        checkpoint_dir: str = "./checkpoints/",
        device: Literal["cuda", "cpu"] = "cuda",
        dev: bool = False,
        validator: Optional[BaseValidator] = None,
    ):
        assert isinstance(max_iter, int)
        assert isinstance(eval_step, int)
        assert callable(metric)
        assert isinstance(checkpoint_dir, (str, Path))
        assert device in ["cuda", "cpu"]
        assert isinstance(dev, bool)
        assert isinstance(validator, BaseValidator) or validator is None

        self.max_iter = max_iter
        self.eval_step = eval_step
        self.metric = metric
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        if validator is None:
            self.validator = BaseValidator(self.metric, is_train=True, device=self.device)
        else:
            self.validator = validator
        if dev:
            self.max_iter = 3
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

    def train(
        self,
        module: nn.Module,
        updater: BaseUpdater,
        *,
        train_dataloader: Optional[DataLoader] = None,
        val_dataloader: Optional[DataLoader] = None,
    ):
        # Set up dataloaders and modules
        self.show_training_info(module, train_dataloader, val_dataloader)

        # Initalize progress bar and tensorboard writer
        train_pbar = tqdm(range(self.max_iter), dynamic_ncols=True)
        writer = SummaryWriter(log_dir=self.checkpoint_dir)

        # Initial stage. Note: updater(module) checks the module and returns a partial func of updating parameters.
        best_metric = 0
        module_update = updater(module)

        for step in train_pbar:
            module.train()

            # Backpropagation
            batch = next(iter(train_dataloader))
            images, masks = batch["image"].to(self.device), batch["label"].to(self.device)
            modality_label = batch["modality"][0]
            assert modality_label in ["ct", "mr"]
            modality = 0 if modality_label == "ct" else 1
            loss = module_update(images, masks, modality)

            # Update progress bar and summary writer
            info = {"step": step + 1, "max_iter": self.max_iter, "modality_label": modality_label, "loss": loss}
            train_pbar.set_description(self.pbar_description.format(**info))
            writer.add_scalar(f"train/{module.criterion.__class__.__name__}", loss, step)

            # Validation
            if ((step + 1) % self.eval_step == 0) or (step == self.max_iter - 1):
                val_metrics = self.validator(module, val_dataloader, global_step=step)

                # Update summary writer
                writer.add_scalar(f"val/{self.metric.__class__.__name__}:Average", val_metrics["mean"], step)
                writer.add_scalar(f"val/{self.metric.__class__.__name__}:CT", val_metrics["ct"], step)
                writer.add_scalar(f"val/{self.metric.__class__.__name__}:MR", val_metrics["mr"], step)

                # Select validation metric
                if min(val_metrics["ct"], val_metrics["mr"]) is not np.nan:
                    val_metric = min(val_metrics["ct"], val_metrics["mr"])
                else:
                    val_metric = val_metrics["mean"]

                # Update best metric
                _green_s = lambda s: "\033[32m" + s + "\033[0m"
                _red_s = lambda s: "\033[31m" + s + "\033[0m"
                if val_metric > best_metric:
                    module.save(self.checkpoint_dir)
                    msg = _green_s(f"Model saved! Validation: (New) {val_metric:2.7f} > (Old) {best_metric:2.7f}")
                    best_metric = val_metric
                else:
                    msg = _red_s(f"No improvement. Validation: (New) {val_metric:2.7f} <= (Old) {best_metric:2.7f}")
                msg += f" (CT) {val_metrics['ct']:2.7f} (MR) {val_metrics['mr']:2.7f}"
                tqdm.write(msg)
