import os
from contextlib import nullcontext
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
import tqdm
from monai.data import DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, Metric
from monai.networks.nets import BasicUNet
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import SGD, Adam, AdamW
from torch.utils.data import ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from lib.loss.target_adaptative_loss import TargetAdaptativeLoss
from lib.utils.validation import get_output_and_mask
from networks.uxnet3d.network_backbone import UXNETDecoder, UXNETEncoder

from .base_trainer import BaseTrainer

torch.backends.cudnn.benchmark = True


class SegmentationModule(nn.Module):
    def __init__(
        self,
        out_channels: int,
        roi_size: tuple,
        sw_batch_size: int,
        feat_extractor: Optional[nn.Module] = None,
        predictor: Optional[nn.Module] = None,
        net: Optional[nn.Module] = None,
        criterion: _Loss = DiceCELoss(to_onehot_y=True, softmax=True),
        optimizer: str = "AdamW",
        lr: float = 0.0001,
        amp=False,
    ):
        super().__init__()

        # A quick way to set up this module is to assign components here:
        net = BasicUNet(
            in_channels=1,
            out_channels=out_channels,
            features=(32, 32, 64, 128, 256, 32),
            spatial_dims=2,
        )
        # feat_extractor = None
        # predictor = None

        self.roi_size = roi_size
        self.sw_batch_size = sw_batch_size
        self.net = net
        self.feat_extractor = feat_extractor
        self.predictor = predictor

        if net:
            params = self.net.parameters()
            if (feat_extractor is not None) or (predictor is not None):
                raise Warning(
                    "net and (feat_extractor, predictor) are both provided. However, only net will be trained."
                )
        elif (feat_extractor is not None) and (predictor is not None):
            params = list(self.feat_extractor.parameters()) + list(self.predictor.parameters())
        else:
            # default network architecture
            self.net = BasicUNet(in_channels=1, out_channels=out_channels, spatial_dims=3)
            params = self.net.parameters()

        differentiable_params = [p for p in params if p.requires_grad]
        self.criterion = criterion
        self.lr = lr
        if optimizer == "AdamW":
            self.optimizer = AdamW(differentiable_params, lr=self.lr)
        if optimizer == "Adam":
            self.optimizer = Adam(differentiable_params, lr=self.lr)
        if optimizer == "SGD":
            self.optimizer = SGD(differentiable_params, lr=self.lr)
        self.amp = amp

    def forward(self, x):
        if self.net:
            y = self.net(x)
        else:
            feature, skip_outputs = self.feat_extractor(x)
            y = self.predictor((feature, skip_outputs))
        return y

    def update(self, x, y, modality=None):
        self.optimizer.zero_grad()
        with torch.autocast(device_type="cuda") if self.amp else nullcontext():
            output = self.forward(x)
            if isinstance(self.criterion, (tuple, list)):
                loss = self.criterion[modality](output, y)
            else:
                loss = self.criterion(output, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def inference(self, x):
        # Using sliding windows
        self.eval()
        return sliding_window_inference(x, self.roi_size, self.sw_batch_size, self.forward)

    def save(self, checkpoint_dir):
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        if self.net:
            torch.save(self.net.state_dict(), os.path.join(checkpoint_dir, "net.pth"))
        else:
            torch.save(self.feat_extractor.state_dict(), os.path.join(checkpoint_dir, "feat_extractor_state.pth"))
            torch.save(self.predictor.state_dict(), os.path.join(checkpoint_dir, "predictor_state.pth"))

    def load(self, checkpoint_dir):
        if self.net:
            self.net.load_state_dict(torch.load(os.path.join(checkpoint_dir, "net.pth")))
        else:
            self.feat_extractor.load_state_dict(torch.load(os.path.join(checkpoint_dir, "feat_extractor_state.pth")))
            self.predictor.load_state_dict(torch.load(os.path.join(checkpoint_dir, "predictor_state.pth")))

    def print_info(self):
        if self.net:
            print("Module:", self.net.__class__.__name__)
        else:
            print("Module Encoder:", self.feat_extractor.__class__.__name__)
            print("       Decoder:", self.predictor.__class__.__name__)
        print("Optimizer:", self.optimizer.__class__.__name__, f"(lr = {self.lr})")
        print("Loss function:", repr(self.criterion))


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
            num_classes=num_classes,
            max_iter=max_iter,
            metric=metric,
            eval_step=eval_step,
            checkpoint_dir=checkpoint_dir,
            device=device,
            data_info=data_info,
            partially_labelled=partially_labelled,
        )


def concat(dataset):
    if dataset[0] and dataset[1]:
        return ConcatDataset(dataset)
    elif dataset[0]:
        return dataset[0]
    elif dataset[1]:
        return dataset[1]
    else:
        raise ValueError("Either index 0 or 1 should be valid dataset.")


class SegmentationInitializer:
    @staticmethod
    def init_dataloaders(train_dataset, val_dataset, test_dataset, batch_size, dev):
        train_dataloader = DataLoader(concat(train_dataset), batch_size=batch_size, shuffle=~dev, pin_memory=True)
        val_dataloader = DataLoader(concat(val_dataset), batch_size=1, shuffle=False, pin_memory=True)
        if any(test_dataset):
            test_dataloader = DataLoader(concat(test_dataset), batch_size=1, shuffle=False, pin_memory=True)
        else:
            test_dataloader = None

        return train_dataloader, val_dataloader, test_dataloader

    @staticmethod
    def init_module(
        out_channels,
        loss,
        optim,
        lr,
        roi_size,
        sw_batch_size,
        data_info,
        modality,
        partially_labelled,
        device,
        **kwargs,
    ):
        if loss != "tal":
            criterion = DiceCELoss(include_background=True, to_onehot_y=True, softmax=True)
        elif modality in ["ct", "mr"]:
            criterion = TargetAdaptativeLoss(data_info["num_classes"], data_info["fg"][modality], device)
        else:
            ct_criterion = TargetAdaptativeLoss(data_info["num_classes"], data_info["fg"]["ct"], device)
            mr_criterion = TargetAdaptativeLoss(data_info["num_classes"], data_info["fg"]["mr"], device)
            criterion = (ct_criterion, mr_criterion)
        module = SegmentationModule(
            out_channels=out_channels,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            optimizer=optim,
            lr=lr,
            criterion=criterion,
            **kwargs,
        )
        return module

    @staticmethod
    def init_trainer(num_classes, max_iter, eval_step, checkpoint_dir, device, data_info, partially_labelled, **kwargs):
        trainer = SegmentationTrainer(
            num_classes=num_classes,
            max_iter=max_iter,
            eval_step=eval_step,
            checkpoint_dir=checkpoint_dir,
            device=device,
            data_info=data_info,
            partially_labelled=partially_labelled,
        )
        return trainer
