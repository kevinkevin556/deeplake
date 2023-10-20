import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import tqdm
from medaset.amos import AMOSDataset
from monai.data import DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.utils import set_determinism
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import SGD, Adam, AdamW
from torch.utils.data import ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from lib.loss.target_adaptive_loss import TargetAdaptiveLoss
from networks.uxnet3d.network_backbone import UXNETDecoder, UXNETEncoder

crop_sample = 2


class SegmentationModule(nn.Module):
    def __init__(
        self,
        feat_extractor=None,
        predictor=None,
        net=None,
        criterion: _Loss = DiceCELoss(to_onehot_y=True, softmax=True),
        optimizer: str = "AdamW",
        lr: float = 0.0001,
        num_classes: int = 16,
    ):
        super().__init__()

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
            # Default feature extractor and predictor
            self.feat_extractor = UXNETEncoder(in_chans=1)
            self.predictor = UXNETDecoder(out_chans=num_classes)
            params = list(self.feat_extractor.parameters()) + list(self.predictor.parameters())

        differentiable_params = [p for p in params if p.requires_grad]
        self.criterion = criterion
        self.lr = lr
        if optimizer == "AdamW":
            self.optimizer = AdamW(differentiable_params, lr=self.lr)
        if optimizer == "Adam":
            self.optimizer = Adam(differentiable_params, lr=self.lr)
        if optimizer == "SGD":
            self.optimizer = SGD(differentiable_params, lr=self.lr)

    def forward(self, x):
        if self.net:
            y = self.net(x)
        else:
            feature, skip_outputs = self.feat_extractor(x)
            y = self.predictor((feature, skip_outputs))
        return y

    def update(self, x, y):
        output = self.forward(x)
        loss = self.criterion(output, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def inference(self, x, roi_size=(96, 96, 96)):
        # Using sliding windows
        self.eval()
        sw_batch_size = crop_sample  # this is used corresponding to amos transforms
        return sliding_window_inference(x, roi_size, sw_batch_size, self.forward)

    def save(self, checkpoint_dir):
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        torch.save(self.feat_extractor.state_dict(), os.path.join(checkpoint_dir, "feat_extractor_state.pth"))
        torch.save(self.predictor.state_dict(), os.path.join(checkpoint_dir, "predictor_state.pth"))

    def load(self, checkpoint_dir):
        self.feat_extractor.load_state_dict(torch.load(os.path.join(checkpoint_dir, "feat_extractor_state.pth")))
        self.predictor.load_state_dict(torch.load(os.path.join(checkpoint_dir, "predictor_state.pth")))

    def print_info(self):
        if self.net:
            print("Module:", self.net.__class__.__name___)
        else:
            print("Module Encoder:", self.feat_extractor.__class__.__name__)
            print("       Decoder:", self.predictor.__class__.__name__)
        print("Optimizer:", self.optimizer.__class__.__name__, f"(lr = {self.lr})")
        print("Loss function:", repr(self.criterion))


class SegmentationTrainer:
    def __init__(
        self,
        max_iter=40000,
        metric=DiceMetric(include_background=True, reduction="mean", get_not_nans=False),
        eval_step=500,
        checkpoint_dir="./default_ckpt/",
        device="cuda",
    ):
        self.max_iter = max_iter
        self.metric = metric
        self.eval_step = eval_step
        self.checkpoint_dir = checkpoint_dir
        self.postprocess = {
            "x": AsDiscrete(argmax=True, to_onehot=AMOSDataset.num_classes),
            "y": AsDiscrete(to_onehot=AMOSDataset.num_classes),
        }
        self.device = device

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

    def validation(self, module, dataloader, global_step=None, **kwargs):
        module.eval()
        val_metrics = []
        val_pbar = tqdm(dataloader, dynamic_ncols=True)
        metric_name = self.metric.__class__.__name__
        train_val_desc = "Validate ({} Steps) ({}={:2.5f})"  # progress bar description used during training
        simple_val_desc = "Validate ({}={:2.5f})"  # progress bar description used when the network is tested
        with torch.no_grad():
            for batch in val_pbar:
                # Infer, decollate data into list of samples, and proprocess both predictions and labels
                images, masks = batch["image"].to(self.device), batch["label"].to(self.device)
                infer_out = module.inference(images)
                samples = decollate_batch({"prediction": infer_out, "ground_truth": masks})
                outputs = [self.postprocess["x"](sample["prediction"]) for sample in samples]
                masks = [self.postprocess["y"](sample["ground_truth"]) for sample in samples]
                # Compute validation metrics
                self.metric(y_pred=outputs, y=masks)
                batch_metric = self.metric.aggregate().item()
                val_metrics.append(batch_metric)
                self.metric.reset()
                # Update progressbar
                if global_step is not None:
                    val_pbar.set_description(train_val_desc.format(global_step, metric_name, batch_metric))
                else:
                    val_pbar.set_description(simple_val_desc.format(metric_name, batch_metric))
        mean_val_metric = np.mean(val_metrics)
        return mean_val_metric

    def train(self, module, train_dataloader, val_dataloader):
        self.show_training_info(module, train_dataloader, val_dataloader)
        best_metric = 0
        train_pbar = tqdm(range(self.max_iter), dynamic_ncols=True)
        writer = SummaryWriter(log_dir=self.checkpoint_dir)
        writer.add_scalar(f"train/{self.metric.__class__.__name__}", 0, 0)  # validation metric starts from zero
        for step in train_pbar:
            module.train()
            batch = next(iter(train_dataloader))
            # Backpropagation
            images, masks = batch["image"].to(self.device), batch["label"].to(self.device)
            loss = module.update(images, masks)
            train_pbar.set_description(f"Training ({step+1} / {self.max_iter} Steps) (loss={loss:2.5f})")
            writer.add_scalar(f"train/{module.criterion.__class__.__name__}", loss, step)
            # Validation
            if ((step + 1) % self.eval_step == 0) or (step == self.max_iter - 1):
                val_metric = self.validation(module, val_dataloader, global_step=step + 1)
                writer.add_scalar(f"train/{self.metric.__class__.__name__}", val_metric, step)
                if val_metric > best_metric:
                    module.save(self.checkpoint_dir)
                    tqdm.write(f"Model saved! Validation: (New) {val_metric:2.7f} > (Old) {best_metric:2.7f}")
                    best_metric = val_metric
                else:
                    tqdm.write(f"No improvement. Validation: (New) {val_metric:2.7f} <= (Old) {best_metric:2.7f}")


class SegmentationInitializer:
    @staticmethod
    def init_dataloaders(train_dataset, val_dataset, test_dataset, batch_size, dev):
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=~dev, pin_memory=True)
        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True)
        test_dataloader = (
            DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True) if test_dataset else None
        )
        return train_dataloader, val_dataloader, test_dataloader

    @staticmethod
    def init_module(loss, optim, lr, data_class, modality, masked, fg, device):
        if loss != "tal":
            criterion = DiceCELoss(include_background=True, to_onehot_y=True, softmax=True)
        elif modality in ["ct", "mr"]:
            criterion = TargetAdaptiveLoss(AMOSDataset.num_classes, fg[modality], device)
        else:
            raise NotImplementedError("Target adaptive loss does not support ct+mr currently.")
        module = SegmentationModule(optimizer=optim, lr=lr, criterion=criterion)
        return module

    @staticmethod
    def init_trainer(max_iter, eval_step, checkpoint_dir, device):
        trainer = SegmentationTrainer(
            max_iter=max_iter,
            eval_step=eval_step,
            checkpoint_dir=checkpoint_dir,
            device=device,
        )
        return trainer
