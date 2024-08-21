# [lenet06] CLI tool

# 在這個教學中，會教你怎麼快速的透過 jsonargpare 打造由 configuration files 控制的命令列工具

# 相較於 notebook，這樣運行你的程式也許比較沒有彈性，但好處是你的參數都可以記錄在一個 yaml 檔案中，便於以後查詢及對照。
# 而 jsonargparse 是 pytorch-lightning 的命令列核心元件，只要在函式或者類別上加入型別提示 (type hint)，
# 他就可以自動幫你生成一個 CLI。(當然你也可以自己寫 parser，更多用法請參考 jsonargparse 的官方說明文件)

# CLI 的相關函式請見 --- CLI
# 一些儲存當下原始程式碼以及設置的函式，請見 --- Utils
# Trainer, Updater 和 Validator 不另外說明。

# 用法：
# 先寫一個 config.yml 或者任意名稱的 yaml file。然後在終端機內輸入
# python "[lenet06] train-cli.py" --config <config.yml>

# python [lenet06] train-cli.py" --help
# 可以印出由 docstring 生成的說明資訊  (需要安裝 jsonargparse[signitures])


from __future__ import annotations

import inspect
import itertools
import os
import shutil
import sys
import warnings
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torchmetrics
import tqdm
from jsonargparse import CLI, ArgumentParser
from jsonargparse.typing import Path_fr
from lenet import LeNet5
from mnist_dataloaders import test_dataloader, train_dataloader, val_dataloader
from monai.data import DataLoader
from monai.metrics import Metric
from ruamel.yaml import YAML
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.append(str(Path.cwd().parent))

from modules.base.trainer import BaseTrainer, TrainLogger
from modules.base.updater import BaseUpdater
from modules.base.validator import BaseValidator


class CustomTrainer(BaseTrainer):
    def __init__(
        self,
        max_iter: int = 10000,
        eval_step: int = 200,
        metric: Metric | None = None,
        validator: BaseValidator | None = None,
        checkpoint_dir: str = "./checkpoints/",
        device: Literal["cuda", "cpu"] = "cuda",
        unpack_item: Callable | Literal["monai", "pytorch"] = "pytorch",
        dev: bool = False,
    ):
        super().__init__(max_iter, eval_step, metric, validator, checkpoint_dir, device, unpack_item, dev)
        self.max_iter = max_iter
        self.eval_step = eval_step
        self.checkpoint_dir = checkpoint_dir
        self.device = device

        self.pbar_description = "Training ({step} / {max_iter} Steps) (loss={loss:2.5f})"

        # Setup validator or metric used during training
        # 設定驗證器
        self.metric = metric
        self.validator = validator if validator else BaseValidator(self.metric, is_train=True, device=self.device)
        self.metric = self.validator.metric

        # The function to unpack the batch into images and targets (based on the __getitem___ of your dataset)
        # 設定打開 batch 為 images 和 targets 的函式
        if unpack_item == "pytorch":
            self.unpack_item = lambda batch: (batch[0].to(self.device), batch[1].to(self.device))
        elif unpack_item == "monai":
            self.unpack_item = lambda batch: (batch["image"].to(self.device), batch["label"].to(self.device))
        else:
            self.unpack_item = unpack_item

        # developer mode
        # 開發模式
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

    def train(
        self,
        module: nn.Module,
        updater: BaseUpdater,
        *,
        train_dataloader: DataLoader | None = None,
        val_dataloader: DataLoader | None = None,
    ):
        self.show_training_info(module, train_dataloader=train_dataloader, val_dataloader=val_dataloader)

        # Initalize progress bar and logger
        # 初始化進度條和紀錄器
        train_pbar = tqdm(range(self.max_iter), dynamic_ncols=True)
        logger = TrainLogger(self.checkpoint_dir)

        # Initial stage. Note: updater(module) checks the module and returns a partial func of updating parameters.
        # 初始化訓練狀態和更新函式
        module.to(self.device)
        best_metric = 0
        module_update = updater(module)

        for step in train_pbar:
            module.train()

            # Backpropagation
            # 反向傳播
            batch = next(iter(train_dataloader))
            images, targets = self.unpack_item(batch)
            loss = module_update(images, targets)  # >> Modified

            # Update progress bar and summary writer
            # 紀錄目前訓練狀態
            info = {"step": step + 1, "max_iter": self.max_iter, "loss": loss}  # >> Modified
            train_pbar.set_description(self.pbar_description.format(**info))
            logger.log_train(module.criterion, loss, step)

            # Validation
            # 驗證目前的網路訓練
            if val_dataloader and ((step + 1) % self.eval_step == 0) or (step == self.max_iter - 1):
                val_metrics = self.validator(module, val_dataloader, global_step=step)
                logger.log_val(self.metric, suffix=["Average"], value=(val_metrics,), step=step)

                # Select validation metric
                # 指定驗證分數
                if val_metrics is not np.nan:
                    val_metric = val_metrics

                # Update best metric
                # 更新驗證分數
                if val_metric > best_metric:
                    module.save(self.checkpoint_dir)
                    logger.success(f"Model saved! Validation: (New) {val_metric:2.5f} > (Old) {best_metric:2.5f}")
                    best_metric = val_metric
                else:
                    logger.info(f"No improvement. Validation: (New) {val_metric:2.5f} <= (Old) {best_metric:2.5f}")


class CustomUpdater(BaseUpdater):
    """Base class of updaters."""

    def __init__(self):
        pass

    def register_module(self, module):
        self.check_module(module)
        # --- Modified
        return partial(self.update, module)

    def update(self, module, images, targets, **kwargs) -> float:
        module.optimizer.zero_grad()
        preds = module(images)
        loss = module.criterion(preds, targets)
        loss.backward()
        module.optimizer.step()
        # --- Modified
        if getattr(module, "scheduler", False):
            module.scheduler.step()
        # --- Modified
        return loss.item()


class CustomValidator(BaseValidator):
    def validation(self, module, dataloader, global_step=None):

        module.eval()
        module.to(self.device)

        if not isinstance(dataloader, (list, tuple)):
            dataloader = [dataloader]
        else:
            dataloader = [dl for dl in dataloader if dl is not None]
        data_iter = itertools.chain(*dataloader)
        pbar = tqdm(
            data_iter,
            total=sum(len(dl) for dl in dataloader),
            dynamic_ncols=True,
        )

        with torch.no_grad():
            for batch in pbar:
                # Infer, decollate data into list of samples, and postprocess both predictions and labels
                images, targets = self.unpack_item(batch)

                # Get inferred / forwarded results of module
                if getattr(module, "inference", False) and self.output_infer:
                    infer_out = module.inference(images)
                else:
                    infer_out = module.forward(images)

                # Compute validation metrics
                batch_metric = self.metric(infer_out, targets).item()  # >> Modified

                # Update progressbar
                info = {
                    "metric_name": self.metric.__class__.__name__,
                    "batch_metric": batch_metric,
                    "global_step": global_step,
                }
                desc = self.pbar_description.format(**info)
                pbar.set_description(desc)

        output = self.metric.compute()  # >> Modified
        self.metric.reset()  # >> Modified
        return output


# --- Utils ---


def save_config_to(dir_path):
    Path(dir_path).mkdir(exist_ok=True, parents=True)
    target_path = os.path.join(dir_path, "config.yml")

    parser = ArgumentParser()
    parser.add_argument("--config", type=Path_fr)
    cfg_path = parser.parse_args().config

    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.preserve_quotes = True
    with open(cfg_path, "r", encoding="utf8") as stream:
        cfg_data = yaml.load(stream)
    with open(target_path, "w", encoding="utf8") as file:
        yaml.dump(cfg_data, file)


def save_source_to(dir_path, objects):
    dir_path = Path(dir_path) / "source"
    dir_path.mkdir(exist_ok=True, parents=True)
    source_files = set(Path(inspect.getsourcefile(obj.__class__)) for obj in objects if obj is not None)
    for file in source_files:
        shutil.copy(file, dir_path / file.name)


# --- CLI ---


# 控制流程
def run(
    name: str = "new-train",
    optimizer: str = "adam",
    lr: float = 0.001,
    scheduler: str | None = None,
    init_method: str = "kaiming",
    checkpoint_dir: str = "./checkpoints",
    max_iter: int = 1000,
):
    """Executes the training and evaluation process for a LeNet5 model using specified parameters.

    Args:
        name: A name identifier for the training session, which is used to create a unique directory for saving checkpoints.
        optimizer: The optimization algorithm to use. Can be either "adam" or "sgd".
        lr: The learning rate for the optimizer.
        scheduler: The learning rate scheduler to use. Can be "cosine" for CosineAnnealingWarmRestarts or None to disable scheduling.
        init_method: The weight initialization method for the model layers. Can be "kaiming" or "xavier".
        checkpoint_dir: The directory where checkpoints will be saved.
        max_iter: The maximum number of training iterations.
    """

    checkpoint_dir = os.path.join(checkpoint_dir, name)
    lenet = LeNet5().cuda()

    # 超參數
    if optimizer == "adam":
        lenet.optimizer = torch.optim.Adam(lenet.net.parameters(), lr=lr)
    if optimizer == "sgd":
        lenet.optimizer = torch.optim.SGD(lenet.net.parameters(), lr=lr)

    if scheduler is None:
        lenet.scheduler = None
    if scheduler == "cosine":
        lenet.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            lenet.optimizer, T_0=10, T_mult=1, eta_min=0, last_epoch=-1
        )

    if init_method == "kaiming":
        pass
    if init_method == "xavier":
        for layer in lenet.net.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)

    # 元件
    validator = CustomValidator(
        metric=torchmetrics.classification.Accuracy(task="multiclass", num_classes=10).to("cuda")
    )
    updater = CustomUpdater()
    trainer = CustomTrainer(
        max_iter=max_iter,
        eval_step=100,
        validator=validator,
        checkpoint_dir=checkpoint_dir,
    )

    # 元件原始程式碼快照
    components = [lenet, trainer, updater, validator]
    save_config_to(trainer.checkpoint_dir)
    save_source_to(trainer.checkpoint_dir, objects=components)

    # 訓練
    print("Train:")
    trainer.train(
        module=lenet,
        updater=updater,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
    )
    print("\n Test:")
    lenet.load(trainer.checkpoint_dir)
    print(validator.validation(module=lenet, dataloader=test_dataloader))


if __name__ == "__main__":
    CLI(run)  # 自動生成 CLI 工具
