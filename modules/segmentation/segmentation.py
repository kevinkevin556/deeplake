from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Union

import numpy as np
import torch
from monai.data import DataLoader as MonaiDataLoader
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, Metric
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import SGD, Adam, AdamW
from torch.utils.data import DataLoader as PyTorchDataLoader

from modules.base.trainer import BaseTrainer
from modules.base.updater import BaseUpdater

DataLoader = Union[MonaiDataLoader, PyTorchDataLoader]


class SegmentationModule(nn.Module):
    alias = "SegNet"

    def __init__(
        self,
        net: nn.Module,
        roi_size: tuple,
        sw_batch_size: int,
        criterion: _Loss = DiceCELoss(to_onehot_y=True, softmax=True),
        optimizer: str = "AdamW",
        lr: float = 0.0001,
        device: Literal["cuda", "cpu"] = "cuda",
        pretrained: Path | None = None,
    ):
        super().__init__()
        self.roi_size = roi_size
        self.sw_batch_size = sw_batch_size
        self.net = net
        self.criterion = criterion
        self.lr = lr
        self.device = device

        params = self.net.parameters()
        differentiable_params = [p for p in params if p.requires_grad]
        # TODO: replace these assignment with partials
        if optimizer == "AdamW":
            self.optimizer = AdamW(differentiable_params, lr=self.lr)
        if optimizer == "Adam":
            self.optimizer = Adam(differentiable_params, lr=self.lr)
        if optimizer == "SGD":
            self.optimizer = SGD(differentiable_params, lr=self.lr)

        if pretrained:
            self.load(pretrained)

        self.to(device)

    def forward(self, x):
        y = self.net(x)
        return y

    def inference(self, x):
        # Using sliding windows
        self.eval()
        return sliding_window_inference(x, self.roi_size, self.sw_batch_size, self.forward)

    def save(self, checkpoint_dir):
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), os.path.join(checkpoint_dir, "net.pth"))

    def load(self, checkpoint_dir):
        self.net.load_state_dict(torch.load(os.path.join(checkpoint_dir, "net.pth")))

    def print_info(self):
        print("Module:", self.net.__class__.__name__)
        print("Optimizer:", self.optimizer.__class__.__name__, f"(lr = {self.lr})")
        print("Loss function:", repr(self.criterion))


class SegmentationEncoderDecoder(nn.Module):
    alias = "SegEncoderDecoder"

    def __init__(
        self,
        feat_extractor: nn.Module,
        predictor: nn.Module,
        roi_size: tuple,
        sw_batch_size: int,
        criterion: _Loss = DiceCELoss(to_onehot_y=True, softmax=True),
        optimizer: str = "AdamW",
        lr: float = 0.0001,
    ):
        super().__init__(self)
        self.roi_size = roi_size
        self.sw_batch_size = sw_batch_size
        self.criterion = criterion
        self.lr = lr

        self.feat_extractor = feat_extractor
        self.predictor = predictor

        params = list(self.feat_extractor.parameters()) + list(self.predictor.parameters())
        differentiable_params = [p for p in params if p.requires_grad]
        # TODO: replace these assignment with partials
        if optimizer == "AdamW":
            self.optimizer = AdamW(differentiable_params, lr=self.lr)
        if optimizer == "Adam":
            self.optimizer = Adam(differentiable_params, lr=self.lr)
        if optimizer == "SGD":
            self.optimizer = SGD(differentiable_params, lr=self.lr)

    def forward(self, x):
        feature, skip_outputs = self.feat_extractor(x)
        y = self.predictor((feature, skip_outputs))
        return y

    def inference(self, x):
        # Using sliding windows
        self.eval()
        return sliding_window_inference(x, self.roi_size, self.sw_batch_size, self.forward)

    def save(self, checkpoint_dir):
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        torch.save(self.feat_extractor.state_dict(), os.path.join(checkpoint_dir, "feat_extractor_state.pth"))
        torch.save(self.predictor.state_dict(), os.path.join(checkpoint_dir, "predictor_state.pth"))

    def load(self, checkpoint_dir):
        self.feat_extractor.load_state_dict(torch.load(os.path.join(checkpoint_dir, "feat_extractor_state.pth")))
        self.predictor.load_state_dict(torch.load(os.path.join(checkpoint_dir, "predictor_state.pth")))

    def print_info(self):
        print("Module Encoder:", self.feat_extractor.__class__.__name__)
        print("       Decoder:", self.predictor.__class__.__name__)
        print("Optimizer:", self.optimizer.__class__.__name__, f"(lr = {self.lr})")
        print("Loss function:", repr(self.criterion))


class SegmentationUpdater(BaseUpdater):
    """A simple updater to update parameters in a segmentation module."""

    alias = "SegUpdater"

    def check_module(self, module):
        assert isinstance(module, torch.nn.Module), "The specified module should inherit torch.nn.Module."
        assert isinstance(
            module, (SegmentationModule, SegmentationEncoderDecoder)
        ), "The specified module should inherit SegmentationModule."
        for component in ("criterion", "optimizer"):
            assert getattr(
                module, component, False
            ), f"The specified module should incoporate component/method: {component}"

    def update(self, module, images, masks, modalities=None):
        module.optimizer.zero_grad()
        output = module.forward(images)
        loss = module.criterion(output, masks)
        loss.backward()
        module.optimizer.step()
        return loss.item()


class ChainShuffleIterator:
    def __init__(self, *dataloaders):
        self.dataloaders = dataloaders
        self.num_dataloaders = len(dataloaders)
        self.total_batches = [len(dl) for dl in self.dataloaders]

    def __len__(self):
        return sum(self.total_batches)

    def __iter__(self):
        self.iterator_list = [iter(dl) for dl in self.dataloaders]
        self.remaining_batches = np.array(self.total_batches.copy())
        return self

    def __next__(self):
        assert sum(self.remaining_batches) >= 0, "Sum of remaining batches should be greater than or equal to 0"

        if np.sum(self.remaining_batches) > 0:
            rng = np.random.default_rng()
            sample_probs = self.remaining_batches / np.sum(self.remaining_batches)
            dataloader_idx = rng.choice(range(self.num_dataloaders), size=1, p=sample_probs)[0]
            self.remaining_batches[dataloader_idx] -= 1
            return next(self.iterator_list[dataloader_idx])
        else:
            raise StopIteration


def chain_shuffle(*dataloaders):
    return ChainShuffleIterator(*dataloaders)


class SegmentationTrainer(BaseTrainer):
    alias = "SegTrainer"

    def __init__(
        self,
        max_iter: int = 10000,
        eval_step: int = 500,
        metric: Metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False),
        checkpoint_dir: str = "./checkpoints/",
        device: Literal["cuda", "cpu"] = "cuda",
        dev: bool = False,
    ):
        super().__init__(max_iter, eval_step, metric, checkpoint_dir, device, dev)

    def train(
        self,
        module,
        updater,
        *,
        train_dataloader: DataLoader | None = None,
        val_dataloader: DataLoader | None = None,
        ct_dataloader: tuple[DataLoader, DataLoader] | None = None,
        mr_dataloader: tuple[DataLoader, DataLoader] | None = None,
    ):
        valid_ct_data = ct_dataloader[0] and ct_dataloader[1]
        valid_mr_data = mr_dataloader[0] and mr_dataloader[1]

        if valid_ct_data and valid_mr_data:
            train_dataloader = chain_shuffle(ct_dataloader[0], mr_dataloader[0])
            val_dataloader = chain_shuffle(ct_dataloader[1], mr_dataloader[1])
        elif valid_ct_data:
            train_dataloader, val_dataloader = ct_dataloader[0], ct_dataloader[1]
        elif valid_mr_data:
            train_dataloader, val_dataloader = mr_dataloader[0], mr_dataloader[1]
        else:
            # train_dataloader = train_dataloader
            # val_dataloader = val_dataloader
            pass

        if train_dataloader is None:
            raise ValueError(
                "No dataloader specified for training."
                "Please provide a valid train_dataloader or both ct_dataloader and mr_dataloader."
            )
        if val_dataloader is None:
            raise ValueError(
                "No dataloader specified for validation."
                "Please provide a valid val_dataloader or both ct_dataloader and mr_dataloader."
            )

        super().train(
            module,
            updater,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
        )
