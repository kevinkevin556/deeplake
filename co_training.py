"""
Implementation based on the following work:

Huang, R., Zheng, Y., Hu, Z., Zhang, S., Li, H. (2020). Multi-organ Segmentation via Co-
training Weight-Averaged Models from Few-Organ Datasets. In: Martel, A.L., et al. Medical
Image Computing and Computer Assisted Intervention – MICCAI 2020. MICCAI 2020. 
Lecture Notes in Computer Science(), vol 12264. Springer, Cham. 
https://doi.org/10.1007/978-3-030-59719-1_15

@author: kevinkevin556
"""

import os
import random
from copy import deepcopy
from itertools import chain
from pathlib import Path
from typing import Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from medaset.transforms import ApplyMaskMappingd, BackgroundifyClassesd
from monai.data import DataLoader, Dataset, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss, DiceLoss, FocalLoss
from monai.metrics import DiceMetric, Metric
from monai.networks.nets import BasicUNet
from monai.transforms import AddChanneld, AsDiscrete, Compose, LoadImaged, ToTensord
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import ConcatDataset, Subset
from torch.utils.tensorboard import SummaryWriter
from torchtyping import TensorType
from tqdm.auto import tqdm

from segmentation import SegmentationModule, SegmentationTrainer


def module_add(self, addend):
    self_module = deepcopy(self)
    with torch.no_grad():
        for self_v, addend_v in zip(self_module.state_dict().values(), addend.state_dict().values()):
            self_v.copy_(self_v + addend_v)
    torch.cuda.empty_cache()
    return self_module


setattr(nn.Module, "__add__", module_add)


def module_mul(self, multiplier):
    self_module = deepcopy(self)
    with torch.no_grad():
        for self_v in self_module.state_dict().values():
            self_v.copy_(self_v * multiplier)
    torch.cuda.empty_cache()
    return self_module


setattr(nn.Module, "__mul__", module_mul)


def get_target_path(data):
    if isinstance(data, (Subset, DataLoader)):
        return get_target_path(data.dataset)
    elif isinstance(data, ConcatDataset):
        output = set(chain.from_iterable([get_target_path(d) for d in data.datasets]))
        return list(output)
    elif isinstance(data, Dataset):
        return data.target_path
    else:
        raise TypeError("Obtained invalid type of dataset when parsing target_path.")


class SoftLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(
        self,
        pred_logits: TensorType["N", "C", "H", "W", "D"],
        lab_logits: TensorType["N", "C", "H", "W", "D"],
        y: TensorType["N", 1, "H", "W", "D"],
    ):
        lab_logits.requires_grad = False
        y.requires_grad = False
        Ty = (y == 0) * 1  # region mask: dim = (N, 1, H, W, D)
        loss = -torch.sum(F.softmax(lab_logits, dim=1) * F.log_softmax(pred_logits, dim=1) * Ty) / torch.sum(Ty)
        return loss


class PretrainedModules:
    def __init__(
        self,
        num_classes: int,
        max_iter: int,
        metric: Metric,
        eval_step: int,
        checkpoint_dir: str,
        device: Literal["cuda", "cpu"],
    ):
        self.num_classes = num_classes
        self.max_iter = max_iter
        self.metric = metric
        self.eval_step = eval_step
        self.checkpoint_dir = checkpoint_dir
        self.device = device

        self.pretrained_module0 = SegmentationModule(
            num_classes=num_classes,
            net=BasicUNet(in_channels=1, out_channels=num_classes, spatial_dims=3),
            criterion=DiceCELoss(to_onehot_y=True, softmax=True),
            optimizer="Adam",
            lr=0.001,
        )
        self.pretrained_module0.to(self.device)

        self.pretrained_module1 = SegmentationModule(
            num_classes=num_classes,
            net=BasicUNet(in_channels=1, out_channels=num_classes, spatial_dims=3),
            criterion=DiceCELoss(to_onehot_y=True, softmax=True),
            optimizer="Adam",
            lr=0.001,
        )
        self.pretrained_module1.to(self.device)

    def train(self, module_id, train_dataloader, val_dataloader):
        if module_id == 0:
            trainer = SegmentationTrainer(
                num_classes=self.num_classes,
                max_iter=self.max_iter,
                metric=self.metric,
                eval_step=self.eval_step,
                checkpoint_dir=os.path.join(self.checkpoint_dir, "pretrained_0"),
                device=self.device,
            )
            trainer.train(self.pretrained_module0, train_dataloader, val_dataloader)
        elif module_id == 1:
            trainer = SegmentationTrainer(
                num_classes=self.num_classes,
                max_iter=self.max_iter,
                metric=self.metric,
                eval_step=self.eval_step,
                checkpoint_dir=os.path.join(self.checkpoint_dir, "pretrained_1"),
                device=self.device,
            )
            trainer.train(self.pretrained_module1, train_dataloader, val_dataloader)
        else:
            raise ValueError(f"Invalid module id. Expect 0 or 1, got {module_id}.")

    def load(self, module_id, checkpoint_dir=None):
        try:
            if module_id == 0:
                if checkpoint_dir is not None:
                    self.pretrained_module0.load(checkpoint_dir)
                else:
                    self.pretrained_module0.load(os.path.join(self.checkpoint_dir, "pretrained_0"))
            elif module_id == 1:
                if checkpoint_dir is not None:
                    self.pretrained_module1.load(checkpoint_dir)
                else:
                    self.pretrained_module1.load(os.path.join(self.checkpoint_dir, "pretrained_1"))
            else:
                raise ValueError(f"Invalid module id. Expect 0 or 1, get {module_id}.")
        except Exception as e:
            raise e
        else:
            print(f"Pretrained module {module_id} is successfully loaded.")

    def get_hard_label(self, image, return_label=0):
        self.pretrained_module0.eval()
        self.pretrained_module1.eval()
        with torch.no_grad():
            label0 = torch.argmax(self.pretrained_module0(image), dim=1, keepdim=True)
            label1 = torch.argmax(self.pretrained_module1(image), dim=1, keepdim=True)
            if return_label == 0:
                return label0 + (label0 == 0) * 1 * label1
            elif return_label == 1:
                return label1 + (label1 == 0) * 1 * label0
            else:
                raise ValueError(f"Invalid value for return_label. Expect {0, 1}, got {return_label}")


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
        lr: float = 0.001,
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
        self.net1 = net_class(in_channels=1, out_channels=num_classes)
        self.net2 = net_class(in_channels=1, out_channels=num_classes)
        self.Enet1 = deepcopy(self.net1)
        self.Enet2 = deepcopy(self.net2)
        for param in self.Enet1.parameters():
            param.requires_grad = False
        for param in self.Enet2.parameters():
            param.requires_grad = False

        # Optimizer
        params = list(self.net1.parameters()) + list(self.net2.parameters())
        self.lr = lr
        if optimizer != "SGD":
            raise UserWarning("No other optimizers than SGD is supported. Use SGD.")
        self.optimizer = SGD(params, lr=self.lr, weight_decay=0.0005, momentum=0.9)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=T_max)

        # Losses
        self.focal_loss = FocalLoss(
            include_background=False,
            to_onehot_y=True,
            gamma=2,
        )
        self.dice_loss = DiceLoss(
            include_background=True,
            to_onehot_y=True,
            softmax=True,
        )
        self.soft_loss = SoftLoss(num_classes)

    def forward(self, x, net=1):
        if net == 1:
            return self.net1(x)
        elif net == 2:
            return self.net2(x)
        else:
            raise ValueError("Invalid net id.")

    def update(
        self,
        x: TensorType["N", "C", "H", "W", "D"],
        y_hat: TensorType["N", 1, "H", "W", "D"],
        y: TensorType["N", 1, "H", "W", "D"],
        lbd_rampup: float = 1,
    ):
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
        self.Enet1 = self.Enet1 * alpha + self.net1 * (1 - alpha)
        self.Enet2 = self.Enet2 * alpha + self.net2 * (1 - alpha)
        soft_loss1 = self.soft_loss(self.net1(x), self.Enet2(x), y)
        soft_loss2 = self.soft_loss(self.net2(x), self.Enet1(x), y)

        # total loss and back-propagation
        total_focal = focal_loss1 + focal_loss2
        total_dice = dice_loss1 + dice_loss2
        total_soft = soft_loss1 + soft_loss2
        total_loss = lbd_focal * total_focal + lbd_dice * total_dice + lbd_rampup * lbd_soft * total_soft
        total_loss.backward()
        self.optimizer.step()
        return total_focal.item(), total_dice.item(), total_soft.item(), total_loss.item()

    def inference(
        self, x: TensorType["N", "C", "H", "W", "D"], roi_size: Tuple[int] = (96, 96, 96), sw_batch_size: int = 2
    ):
        # Using sliding windows
        self.eval()
        return sliding_window_inference(x, roi_size, sw_batch_size, self.forward)

    def save(self, checkpoint_dir: str):
        torch.save(self.net1.state_dict(), os.path.join(checkpoint_dir, "net1.pth"))
        torch.save(self.net2.state_dict(), os.path.join(checkpoint_dir, "net2.pth"))

    def load(self, checkpoint_dir: str):
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
        print(
            f"Loss function: {self.focal_loss.__class__.__name__}, "
            + f"{self.dice_loss.__class__.__name__}, "
            + f"{self.soft_loss.__class__.__name__}"
        )


class CoTrainingTrainer:
    def __init__(
        self,
        num_classes: int,
        max_iter: int = 10000,
        metric: Metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False),
        eval_step: int = 100,
        checkpoint_dir: str = "./checkpoints/",
        device: Literal["cuda", "cpu"] = "cuda",
        pretrained_dir: str = None,
    ):
        self.num_classes = num_classes
        self.max_iter = max_iter
        self.metric = metric
        self.eval_step = eval_step
        self.checkpoint_dir = checkpoint_dir
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        self.device = device
        self.pretrained_modules = PretrainedModules(
            num_classes=num_classes,
            max_iter=max_iter,
            metric=DiceMetric(),
            eval_step=100,
            checkpoint_dir=pretrained_dir if pretrained_dir else checkpoint_dir,
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
        print("# of Training Samples:", {"ct": len(train_dataloader[0]), "mr": len(train_dataloader[1])})
        print("# of Validation Samples:", {"ct": len(val_dataloader[0]), "mr": len(val_dataloader[1])})
        print("Max iteration:", self.max_iter, f"steps (validates per {self.eval_step} steps)")
        print("Checkpoint directory:", self.checkpoint_dir)
        print("Evaluation metric:", self.metric.__class__.__name__)
        module.print_info()
        print("--------")

    def pretrain_single_organ(self, module_id, train_dataloader, val_dataloader):
        assert module_id in [0, 1], f"Invalid module id. Got {module_id}, expect {0, 1}"
        self.pretrained_modules.train(module_id, train_dataloader, val_dataloader)
        self.pretrained_modules.load(module_id)
        torch.cuda.empty_cache()
        if module_id == 0:
            for param in self.pretrained_modules.pretrained_module0.parameters():
                param.requires_grad = False
        else:
            for param in self.pretrained_modules.pretrained_module1.parameters():
                param.requires_grad = False

    def cotrain_multi_organ(self, module, train_dataloader, val_dataloader):
        best_metric = 0
        train_pbar = tqdm(range(self.max_iter), dynamic_ncols=True)
        writer = SummaryWriter(log_dir=self.checkpoint_dir)
        writer.add_scalar(f"train/{self.metric.__class__.__name__}", best_metric, 0)

        iter_count = 0
        train_iterator = chain(*train_dataloader)  # Concatenate two iterables (dataloaders) into one
        for step in train_pbar:
            ## Train
            module.train()
            try:
                batch = next(train_iterator)
            except StopIteration:
                # If the iterator runs out of data, create a new one.
                iter_count = 0
                train_iterator = chain(*train_dataloader)
                batch = next(train_iterator)
            # If the counter of iterator is smaller than the length of dataloader0
            # it means the sample is from the first dataset.
            # The hard label prediction should be obatained mainly from the first pretrained model.
            image, mask = batch["image"].to(self.device), batch["label"].to(self.device)
            mask_hat = self.pretrained_modules.get_hard_label(
                image,
                return_label=(0 if iter_count < len(train_dataloader[0]) else 1),
            )
            iter_count += 1
            # There is no detail description about how lambda_rampup is gradually increased during training,
            # so I apply a similar increasing strategy as the I do to the GRL weight.
            p = float(step) / self.max_iter
            q = 2.0 / (1.0 + np.exp(-10 * p)) - 1
            focal, dice, soft, total = module.update(x=image, y_hat=mask_hat, y=mask, lbd_rampup=q)
            train_pbar.set_description(
                f"Training ({step+1} / {self.max_iter} Steps) (focal={focal:.4e}, dice={dice:.4e}, soft={soft:.4e}, total={total:.4e})"
            )
            writer.add_scalar(f"train/focal_loss", focal, step)
            writer.add_scalar(f"train/dice_loss", dice, step)
            writer.add_scalar(f"train/soft_loss", soft, step)
            writer.add_scalar(f"train/total_loss", total, step)
            ## Validation
            if (step + 1) % self.eval_step == 0 or (step + 1) == self.max_iter:
                val_metric = self.validation(module, val_dataloader, step)
                writer.add_scalar(f"train/{self.metric.__class__.__name__}", val_metric, step)
                if val_metric > best_metric:
                    module.save(self.checkpoint_dir)
                    tqdm.write(f"Model saved! Validation: (New) {val_metric:2.7f} > (Old) {best_metric:2.7f}")
                    best_metric = val_metric
                else:
                    tqdm.write(f"No improvement. Validation: (New) {val_metric:2.7f} <= (Old) {best_metric:2.7f}")

    def get_organ_pixels(self, dataloader):
        # Obtain dataset info
        data_class = dataloader[0].dataset.dataset.__class__
        target_path = set(get_target_path(dataloader[0]) + get_target_path(dataloader[1]))
        excluded_classes = getattr(data_class, "excluded_classes", None)
        relabelling = getattr(data_class, "relabelling", None)
        # Create loading transform
        transforms = [
            LoadImaged(keys=["label"], image_only=True, dtype=np.uint8),
            AddChanneld(keys=["label"]),
        ]
        if excluded_classes:
            transforms.append(BackgroundifyClassesd(keys=["label"], classes=excluded_classes))
        if relabelling:
            transforms.append(ApplyMaskMappingd(keys=["label"], mask_mapping=relabelling))
        transforms.append(ToTensord(keys=["label"], dtype=torch.uint8, device=self.device))
        # Create dataset to read labels and count pixels for each organ
        temp_dataset = Dataset(data=[{"label": p} for p in target_path], transform=Compose(transforms))
        organ_pixel_count = torch.zeros(self.num_classes).to(self.device)
        for d in tqdm(temp_dataset, desc="Counting pixels of each organ ..."):
            organ_pixel_count += torch.bincount(torch.flatten(d["label"]))
        return organ_pixel_count

    def train(self, module, train_dataloader, val_dataloader, load_pretrained: Optional[Union[list, dict]] = None):
        ## Load or pretrain modules
        for p_id in [0, 1]:
            if isinstance(load_pretrained, dict) and p_id in load_pretrained.keys():
                self.pretrained_modules.load(p_id, load_pretrained[p_id])
            elif isinstance(load_pretrained, (list, tuple)) and p_id in load_pretrained:
                self.pretrained_modules.load(p_id)
            else:
                self.pretrain_single_organ(p_id, train_dataloader[p_id], val_dataloader[p_id])

        ## Train with cotraining module
        self.show_training_info(module, train_dataloader, val_dataloader)
        # Configure focal loss in cotraining module
        w = self.get_organ_pixels(train_dataloader)
        w_inv = w[1:] ** (-1)
        focal_alpha = w_inv / torch.sum(w_inv)
        module.focal_loss = FocalLoss(
            include_background=False,
            to_onehot_y=True,
            gamma=2,
            weight=focal_alpha,
        )
        self.cotrain_multi_organ(module, train_dataloader, val_dataloader)

    def validation(self, module, dataloader, global_step=None, **kwargs):
        module.eval()
        val_metrics = []
        val_pbar = tqdm(
            iterable=chain(*dataloader),
            total=sum([len(d) for d in dataloader]),
            dynamic_ncols=True,
        )
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
    def init_module(loss, optim, lr, dataset, modality, masked, device, **kwargs):
        module = CoTrainingModule(num_classes=dataset["num_classes"], lr=lr, **kwargs)
        return module

    @staticmethod
    def init_trainer(num_classes, max_iter, eval_step, checkpoint_dir, device, pretrained_dir=None):
        trainer = CoTrainingTrainer(
            num_classes=num_classes,
            max_iter=max_iter,
            eval_step=eval_step,
            checkpoint_dir=checkpoint_dir,
            device=device,
            pretrained_dir=pretrained_dir,
        )
        return trainer
