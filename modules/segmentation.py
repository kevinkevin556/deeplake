import os
from pathlib import Path
from typing import Final, Literal, Optional

import numpy as np
import torch
import tqdm
from monai.data import DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, Metric
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import SGD, Adam, AdamW
from torch.utils.data import ConcatDataset, Dataset

from lib.loss.target_adaptative_loss import TargetAdaptativeLoss
from lib.utils.validation import get_output_and_mask
from networks.uxnet3d.network_backbone import UXNETDecoder, UXNETEncoder

from .base_trainer import BaseTrainer

torch.backends.cudnn.benchmark = True


class SegmentationModule(nn.Module):
    def __init__(
        self,
        net: nn.Module,
        roi_size: tuple,
        sw_batch_size: int,
        criterion: _Loss = DiceCELoss(to_onehot_y=True, softmax=True),
        optimizer: str = "AdamW",
        lr: float = 0.0001,
        device: Literal["cuda", "cpu"] = "cuda",
    ):
        super().__init__()
        self.roi_size = roi_size
        self.sw_batch_size = sw_batch_size
        self.net = net
        self.criterion = criterion
        self.lr = lr
        self.device = device

        self.net.to(device)

        params = self.net.parameters()
        differentiable_params = [p for p in params if p.requires_grad]
        # TODO: replace these assignment with partials
        if optimizer == "AdamW":
            self.optimizer = AdamW(differentiable_params, lr=self.lr)
        if optimizer == "Adam":
            self.optimizer = Adam(differentiable_params, lr=self.lr)
        if optimizer == "SGD":
            self.optimizer = SGD(differentiable_params, lr=self.lr)

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

    def setup(self, loss: str, modality: str, data_info: dict, *args, **kwargs):
        if loss != "tal":
            self.criterion = DiceCELoss(include_background=True, to_onehot_y=True, softmax=True)
        elif modality in ["ct", "mr"]:
            self.criterion = TargetAdaptativeLoss(data_info["num_classes"], data_info["fg"][modality], self.device)
        else:
            ct_criterion = TargetAdaptativeLoss(data_info["num_classes"], data_info["fg"]["ct"], self.device)
            mr_criterion = TargetAdaptativeLoss(data_info["num_classes"], data_info["fg"]["mr"], self.device)
            self.criterion = (ct_criterion, mr_criterion)


class SegmentationEncoderDecoder(nn.Module):
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
        super().__init__()
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

    def setup(self, loss: str, modality: str, data_info: dict, *args, **kwargs):
        if loss != "tal":
            self.criterion = DiceCELoss(include_background=True, to_onehot_y=True, softmax=True)
        elif modality in ["ct", "mr"]:
            self.criterion = TargetAdaptativeLoss(data_info["num_classes"], data_info["fg"][modality], self.device)
        else:
            ct_criterion = TargetAdaptativeLoss(data_info["num_classes"], data_info["fg"]["ct"], self.device)
            mr_criterion = TargetAdaptativeLoss(data_info["num_classes"], data_info["fg"]["mr"], self.device)
            self.criterion = (ct_criterion, mr_criterion)


class SegmentationUpdater:
    """A simple updater to update parameters in a segmentation module."""

    def check_module(self, module):
        assert isinstance(module, torch.nn.Module), "The specified module should inherit torch.nn.Module."
        assert isinstance(
            module, (SegmentationModule, SegmentationEncoderDecoder)
        ), "The specified module should inherit SegmentationModule."
        for component in ["criterion", "optimizer"]:
            assert getattr(
                module, component, False
            ), "The specified module should incoporate component/method: {component}"

    def update(self, module, images, masks, modalities):
        module.optimizer.zero_grad()
        output = module.forward(images)
        loss = module.criterion(output, masks)
        loss.backward()
        module.optimizer.step()
        return loss.item()


def concat(dataset):
    if dataset[0] and dataset[1]:
        return ConcatDataset(dataset)
    elif dataset[0]:
        return dataset[0]
    elif dataset[1]:
        return dataset[1]
    else:
        raise ValueError("Either index 0 or 1 should be valid dataset.")


class SegmentationTrainer(BaseTrainer):
    def __init__(
        self,
        num_classes: int,
        max_iter: int = 10000,
        metric: Metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False),
        eval_step: int = 500,
        checkpoint_dir: str = "./checkpoints/",
        device: Literal["cuda", "cpu"] = "cuda",
        data_info: dict = None,
        partially_labelled: bool = False,
    ):
        super().__init__(
            num_classes, max_iter, metric, eval_step, checkpoint_dir, device, data_info, partially_labelled
        )

    def setup_data(self, train_data, val_data, batch_size, dev, *args, **kwargs):
        if isinstance(train_data, Dataset):
            self.train_dataloader = DataLoader(concat(train_data), batch_size=batch_size, shuffle=~dev, pin_memory=True)
        else:
            self.train_dataloader = train_data

        if isinstance(val_data, Dataset):
            self.val_dataloader = DataLoader(concat(val_data), batch_size=1, shuffle=False, pin_memory=True)
        else:
            self.val_dataloader = val_data

    def train(self, module, updater, train_data, val_data, *args, **kwargs):
        super().train(module, updater, train_data, val_data, *args, **kwargs)
