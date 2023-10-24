import os
import random
from itertools import chain
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from monai.data import DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss, DiceLoss, FocalLoss
from monai.metrics import DiceMetric, Metric
from monai.networks.nets import BasicUNet
from monai.networks.utils import one_hot
from monai.transforms import AsDiscrete
from torch import nn
from torch.optim import SGD, Adam, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from segmentation import SegmentationModule, SegmentationTrainer


class SoftLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, pred_logits, lab_logits, y):
        Ty = 1 - one_hot(y, self.num_classes)
        loss = -torch.sum(Ty * F.softmax(lab_logits, dim=1) * F.log_softmax(pred_logits, dim=1))
        return loss


class PretrainedModules:
    def __init__(self, max_iter, metric, eval_step, checkpoint_dir, device):
        self.max_iter = max_iter
        self.metric = metric
        self.eval_step = eval_step
        self.checkpoint = checkpoint_dir
        self.device = device

        self.pretrained_module0 = SegmentationModule(
            net=BasicUNet(in_channels=1, out_channels=2, spatial_dims=3),
            criterion=DiceCELoss(to_onehot=True, softmax=True),
            optimizer="Adam",
            lr=0.001,
        )
        self.pretrained_module1 = SegmentationModule(
            net=BasicUNet(in_channels=1, out_channels=2, spatial_dims=3),
            criterion=DiceCELoss(to_onehot=True, softmax=True),
            optimizer="Adam",
            lr=0.001,
        )

    def train(self, module_id, train_dataloader, val_dataloader):
        if module_id == 0:
            trainer = SegmentationTrainer(
                max_iter=self.max_iter,
                metric=self.metric,
                eval_step=self.eval_step,
                checkpoint=os.path.join(self.checkpoint_dir, "pretrained_0"),
                device=self.device,
            )
            trainer.train(self.pretrained_module0, train_dataloader, val_dataloader)
        elif module_id == 1:
            trainer = SegmentationTrainer(
                max_iter=self.max_iter,
                metric=self.metric,
                eval_step=self.eval_step,
                checkpoint=os.path.join(self.checkpoint_dir, "pretrained_1"),
                device=self.device,
            )
            self.trainer.train(self.pretrained_module1, train_dataloader, val_dataloader)
        else:
            raise ValueError(f"Invalid module id. Expect 0 or 1, got {module_id}.")

    def load(self, module_id):
        try:
            if module_id == 0:
                self.pretrained_module0.load(os.path.join(self.checkpoint_dir, "pretrained_0"))
            elif module_id == 1:
                self.pretrained_module1.load(os.path.join(self.checkpoint_dir, "pretrained_1"))
            else:
                raise ValueError(f"Invalid module id. Expect 0 or 1, get {module_id}.")
        except Exception as e:
            raise e

    def get_hard_label(self, image):
        label0 = torch.argmax(self.pretrained_module0(image))
        label1 = torch.argmax(self.pretrained_module1(image))
        # Because we do not know which label is better
        # randomly pick one as the base mask and fill background pixels with the other
        if random.randint(0, 1) == 0:
            return label0 + label1[label0 == 0]
        else:
            return label1 + label0[label1 == 0]


class CoTrainingModule(nn.Module):
    def __init__(
        self,
        num_classes: int,
        net_class: nn.Module = BasicUNet,
        alpha: float = 0.999,
        lambda_focal: float = 1.0,
        lambda_dice: float = 0.1,
        lambda_soft: float = 0.1,
        optimizer: str = "SGD",
        lr: float = 0.05,
        T_max: int = 10000,
    ):
        super().__init__()

        # hyperparameters
        self.alpha = alpha
        self.lbd_focal = lambda_focal
        self.lbd_dice = lambda_dice
        self.lbd_soft = lambda_soft

        # Network components
        self.num_classes = num_classes  # number of AMOS classes
        self.pretrained_net1 = None
        self.pretrained_net2 = None
        self.net1 = net_class(in_channels=1, out_channels=num_classes)
        self.net2 = net_class(in_channels=1, out_channels=num_classes)

        # # Optimizer
        params = list(self.net1.parameters()) + list(self.net2.parameters())
        self.lr = lr
        if optimizer == "Adam":
            self.optimizer = Adam(params, lr=self.lr)
        if optimizer == "SGD":
            self.optmizer = SGD(params, lr=self.lr, weight_decay=0.0005, momentum=0.9)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=T_max)

        # Losses
        self.focal_loss = FocalLoss(include_background=True)
        self.dice_loss = DiceLoss(include_background=True)
        self.soft_loss = SoftLoss(num_classes)

    def forward(self, x, net=1):
        if net == 1:
            return self.net1(x)
        elif net == 2:
            return self.net2(x)
        else:
            raise ValueError("Invalid net id.")

    def update(self, x, y_hat, y, lbd_rampup=1):
        self.optimizer.zero_grad()

        # Parameters
        alpha = self.alpha
        lbd_focal = self.lbd_focal
        lbd_dice = self.lbd_dice
        lbd_soft = self.lbd_soft

        # hard pseudo label
        focal_loss1 = self.focal_loss(self.net1(x), y_hat)
        dice_loss1 = self.dice_loss(self.net1(x), y_hat)
        focal_loss2 = self.focal_loss(self.net2(x), y_hat)
        dice_loss2 = self.dice_loss(self.net2(x), y_hat)

        # soft pseudo label
        self.Enet1 = alpha * self.Enet1 + (1 - alpha) * self.net1
        self.Enet2 = alpha * self.Enet2 + (1 - alpha) * self.net2
        soft_loss1 = self.soft_loss(self.net1(x), self.Enet2(x), y)
        soft_loss2 = self.soft_loss(self.net2(x), self.Enet1(x), y)

        # total loss and back-propagation
        total_loss = (
            lbd_focal * (focal_loss1 + focal_loss2)
            + lbd_dice * (dice_loss1 + dice_loss2)
            + lbd_rampup * lbd_soft * (soft_loss1 + soft_loss2)
        )
        total_loss.backward()
        self.optimizer.step()
        return total_loss.item()

    def inference(self, x, roi_size=(96, 96, 96), sw_batch_size=2):
        # # Using sliding windows
        self.eval()
        return sliding_window_inference(x, roi_size, sw_batch_size, self.forward)

    def save(self, checkpoint_dir):
        torch.save(self.net1.state_dict(), os.path.join(checkpoint_dir, "net1.pth"))
        torch.save(self.net2.state_dict(), os.path.join(checkpoint_dir, "net2.pth"))

    def load(self, checkpoint_dir):
        try:
            self.net1.load_state_dict(torch.load(os.path.join(checkpoint_dir, "net1.pth")))
            self.net2.load_state_dict(torch.load(os.path.join(checkpoint_dir, "net2.pth")))
        except Exception as e:
            raise e
        pass

    def print_info(self):
        print("Module Net-1:", self.net1.__class__.__name__)
        print("       Net-2:", self.net2.__class__.__name__)
        print("Optimizer:", self.optimizer.__class__.__name__, f"(lr = {self.lr})")
        print("Loss function:", {"ct": self.ct_tal.__class__.__name__, "mr": self.mr_tal.__class__.__name__})


class CoTrainingTrainer:
    def __init__(
        self,
        num_classes: int,
        max_iter: int = 10000,
        metric: Metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False),
        eval_step: int = 100,
        checkpoint_dir: str = "./checkpoints/",
        device: Literal["cuda", "cpu"] = "cuda",
    ):
        self.max_iter = max_iter
        self.metric = metric
        self.eval_step = eval_step
        self.checkpoint_dir = checkpoint_dir
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        self.device = device
        self.pretrained_modules = PretrainedModules(
            max_iter=10000,
            metric=DiceMetric(),
            eval_step=100,
            checkpoint_dir=checkpoint_dir,
            device=device,
        )
        self.postprocess = {
            "x": AsDiscrete(argmax=True, to_onehot=num_classes),
            "y": AsDiscrete(to_onehot=num_classes),
        }

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

    def pretrain_single_organ(self, module_id, train_dataloader, val_dataloader):
        self.pretrained_modules.train(module_id, train_dataloader, val_dataloader)
        self.pretrained_modules.load(module_id)

    def cotrain_multi_organ(self, module, train_dataloader, val_dataloader):
        best_metric = 0
        train_pbar = tqdm(range(self.max_iter), dynamic_ncols=True)
        writer = SummaryWriter(log_dir=self.checkpoint_dir)
        writer.add_scalar(f"train/{self.metric.__class__.__name__}", 0, 0)  # validation metric starts from zero

        for step in range(self.max_iter):
            module.train()
            # Train
            batch = next(iter(train_dataloader))
            image, mask = batch["image"].to(self.device), batch["label"].to(self.device)
            mask_hat = self.pretrained_modules.get_hard_label(image)
            loss = module.update(image, mask_hat, mask)
            train_pbar.set_description(f"Training ({step+1} / {self.max_iter} Steps) (loss={loss:2.5f})")
            writer.add_scalar(f"train/{module.criterion.__class__.__name__}", loss, step)

            # Validation
            if (step + 1) % self.eval_step == 0 or (step + 1) == self.max_iter:
                val_metric = self.validation(module, val_dataloader, step)
                if val_metric > best_metric:
                    module.save(self.checkpoint_dir)
                    tqdm.write(f"Model saved! Validation: (New) {val_metric:2.7f} > (Old) {best_metric:2.7f}")
                    best_metric = val_metric
                else:
                    tqdm.write(f"No improvement. Validation: (New) {val_metric:2.7f} <= (Old) {best_metric:2.7f}")

    def train(self, module, train_dataloader, val_dataloader, not_pretrained=[0, 1]):
        for i in not_pretrained:
            self.pretrain_single_organ(i, train_dataloader[i], val_dataloader[i])

        self.show_training_info(module, train_dataloader, val_dataloader)
        self.cotrain_multi_organ(module, train_dataloader, val_dataloader)

    def validation(self, module, dataloader, global_step=None, **kwargs):
        module.eval()
        val_metrics = []
        val_pbar = tqdm(chain(**dataloader), dynamic_ncols=True)
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


class CoTrainingInitializer:
    @staticmethod
    def init_dataloaders(train_dataset, val_dataset, test_dataset, batch_size, dev):
        train_dataloaders = tuple(
            [DataLoader(train_dataset[i], batch_size=batch_size, shuffle=~dev, pin_memory=True) for i in range(2)]
        )
        val_dataloaders = tuple(
            [DataLoader(val_dataset[i], batch_size=1, shuffle=False, pin_memory=True) for i in range(2)]
        )
        test_dataloaders = tuple(
            [DataLoader(test_dataset[i], batch_size=1, shuffle=False, pin_memory=True) for i in range(2)]
        )
        return train_dataloaders, val_dataloaders, test_dataloaders

    @staticmethod
    def init_module(loss, optim, lr, dataset, modality, masked, device):
        module = CoTrainingModule(num_classes=dataset["num_classes"], lr=lr)
        return module

    @staticmethod
    def init_trainer(num_classes, max_iter, eval_step, checkpoint_dir, device):
        trainer = CoTrainingTrainer(
            num_classes=num_classes,
            max_iter=max_iter,
            eval_step=eval_step,
            checkpoint_dir=checkpoint_dir,
            device=device,
        )
        return trainer
