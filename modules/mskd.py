import argparse
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
from monai.networks.utils import one_hot
from monai.transforms import AddChanneld, AsDiscrete, Compose, LoadImaged, ToTensord
from torch import nn, ones, zeros
from torch.nn import BCEWithLogitsLoss
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import ConcatDataset, Subset
from torch.utils.tensorboard import SummaryWriter
from torchtyping import TensorType
from tqdm import tqdm
from tqdm.auto import tqdm


# Type Annotation
class NCHWD(TensorType):
    def __init__(self):
        super().__init__()
        super().__class_getitem__(["N", "C", "H", "W", "D"])


class N1HWD:
    def __init__(self):
        super().__init__()
        super().__class_getitem__(["N", 1, "H", "W", "D"])


class Nchwd(TensorType):
    def __init__(self):
        super().__init__()
        super().__class_getitem__(["N", "C1", "H1", "W1", "D1"])


class N1hwd:
    def __init__(self):
        super().__init__()
        super().__class_getitem__(["N", 1, "H1", "W1", "D1"])


# Utils
def map_supervision_signal(
    logit: TensorType["NCHWD"],
    bg: list,
):
    # A modified vesion of (3) in Feng et al. (2021) which
    # requires the input logit to contain the exact number of
    # channels as the number of clasees in the fully-annotated
    # dataset.
    q: NCHWD = F.softmax(logit, dim=1)
    q[:, bg, ...] = 0
    ch_sum: N1HWD = torch.sum(q, dim=1, keepdim=True)
    q_hat: NCHWD = q / ch_sum
    return q_hat


# Modules
class MSKDModule(nn.Module):
    def __init__(self, num_classes: int, background: tuple):
        super.__init__()
        self.num_classes = num_classes
        self.background0, self.background1 = background

        self.teacher0 = BasicUNet(in_channels=1, out_channels=num_classes)
        self.teacher1 = BasicUNet(in_channels=1, out_channels=num_classes)
        self.student = BasicUNet(in_channels=1, out_channels=num_classes)
        self.kl_loss = nn.KLDivLoss(reduction="none")
        self.optimizer = Adam(self.student.parameters(), lr=0.001)

        self.feat_stud = None
        self.feat0 = None
        self.feat1 = None

        # The hook is configured based on the BasicUNet Architecture
        self.teacher0.upcat_1.convs.conv_1.register_hook(self.get_activation("teacher0"))
        self.teacher1.upcat_1.convs.conv_1.register_hook(self.get_activation("teacher1"))
        self.student.upcat_1.convs.conv_1.register_hook(self.get_activation("student"))

    def get_activation(self, net: str):
        def hook(model, input, output):
            if net == "teacher0":
                self.feat0 = output
            elif net == "teacher1":
                self.feat1 = output
            elif net == "student":
                self.feat_stud = output
            else:
                raise ValueError("Invalid net name.")

        return hook

    def forward(self, x):
        return self.student(x)

    def update(self, x):
        bg0 = self.background0
        bg1 = self.background1
        lbd1 = self.lambda_background
        lbd2 = self.lambda_feature

        # Logit-wise supervision
        logit_stud: NCHWD = self.student(x)
        log_prob_stud: NCHWD = F.log_softmax(logit_stud, dim=1)
        n, c, h, w, d = logit_stud.shape

        ## teacher 0 & teacher 1
        logit0: NCHWD = self.teacher0(x)
        prob0: NCHWD = map_supervision_signal(logit0, bg0)
        M0: N1HWD = (torch.argmax(prob0, dim=1, keepdim=True) > 0) * 1
        logit_loss0: float = torch.sum(self.kl_loss(log_prob_stud, prob0) * M0) / (n * h * w * d)

        logit1: NCHWD = self.teacher1(x)
        prob1: NCHWD = map_supervision_signal(logit1, bg1)
        M1: N1HWD = (torch.argmax(prob1, dim=1, keepdim=True) > 0) * 1
        logit_loss1: float = torch.sum(self.kl_loss(log_prob_stud, prob1) * M1) / (n * h * w * d)

        logit_loss = logit_loss0 + logit_loss1

        ## background
        Mb: N1HWD = (1 - M0) * (1 - M1)
        prob_b: NCHWD = (prob0 + prob1) / 2
        logit_loss_b: float = torch.sum(self.kl_loss(log_prob_stud, prob_b) * Mb) / (n * h * w * d)

        # Feature-wise supervision (l=1 in decoder)
        feat_stud: Nchwd = torch.sort(self.feat_stud, dim=1)
        n, c1, h1, w1, d1 = feat_stud.shape

        M0_layer: N1hwd = F.max_pool3d(M0, kernel_size=2)
        feat0: Nchwd = torch.sort(self.feat0, dim=1)
        feature_loss0: float = torch.sum(self.kl_loss(torch.log(feat_stud), feat0) * M0_layer) / (n * h1 * w1 * d1)

        M1_layer: N1hwd = F.max_pool3d(M1, kernel_size=2)
        feat1: Nchwd = torch.sort(feat1, dim=1)
        feature_loss1: float = torch.sum(self.kl_loss(torch.log(feat_stud), feat1) * M1_layer) / (n * h1 * w1 * d1)

        feature_loss = feature_loss0 + feature_loss1
        total_loss = logit_loss + lbd1 * logit_loss_b + lbd2 * feature_loss

        # Backpropagation
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss.item()

    def inference(self, x, roi_size=(96, 96, 96), sw_batch_size=2):
        # Using sliding windows
        self.eval()
        return sliding_window_inference(x, roi_size, sw_batch_size, self.forward)

    def save(self, checkpoint_dir):
        torch.save(self.teacher0.state_dict(), os.path.join(checkpoint_dir, "teacher0.pth"))
        torch.save(self.teacher1.state_dict(), os.path.join(checkpoint_dir, "teacher1.pth"))
        torch.save(self.student.state_dict(), os.path.join(checkpoint_dir, "student.pth"))

    def load(self, checkpoint_dir):
        try:
            self.teacher0.load_state_dict(torch.load(os.path.join(checkpoint_dir, "teacher0_state.pth")))
            self.teacher1.load_state_dict(torch.load(os.path.join(checkpoint_dir, "teacher1_state.pth")))
            self.student.load_state_dict(torch.load(os.path.join(checkpoint_dir, "student_state.pth")))
        except Exception as e:
            raise e

    def print_info(self):
        print("Module Teacher-0:", self.teacher0.__class__.__name__)
        print("       Teacher-1:", self.teacher1.__class__.__name__)
        print("       Student:", self.student.__class__.__name__)
        print("Optimizer:", self.optimizer.__class__.__name__, f"(lr = {self.lr})")
        print(f"Loss function: {self.kl_loss.__class__.__name__}")


class MSKDTrainer:
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

    def distillate_teachers_into_student(self, module, train_dataloader, val_dataloader):
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
            image, mask = batch["image"].to(self.device), batch["label"].to(self.device)
            train_loss = module.update(x=image)
            train_pbar.set_description(f"Training ({step+1} / {self.max_iter} Steps) (loss = {train_loss:2.5f})")
            writer.add_scalar(f"train/train_loss", train_loss, step)

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

    def train(self, module, train_dataloader, val_dataloader, load_pretrained: Optional[Union[list, dict]] = None):
        ## Load or pretrain modules
        for p_id in [0, 1]:
            if isinstance(load_pretrained, dict) and p_id in load_pretrained.keys():
                self.pretrained_modules.load(p_id, load_pretrained[p_id])
            elif isinstance(load_pretrained, (list, tuple)) and p_id in load_pretrained:
                self.pretrained_modules.load(p_id)
            else:
                self.pretrain_single_organ(p_id, train_dataloader[p_id], val_dataloader[p_id])

        # Assign pretrained modules as teachers in mskd module
        module.teacher0.load_state_dict(self.pretrained_modules.pretrained_module0.net.state_dict())
        module.teacher1.load_state_dict(self.pretrained_modules.pretrained_module1.net.state_dict())

        ## Train with mskd module
        self.show_training_info(module, train_dataloader, val_dataloader)
        self.distillate_teachers_into_student(module, train_dataloader, val_dataloader)

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


class MSKDInitializer:
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
        module = MSKDModule(num_classes=dataset["num_classes"], lr=lr, **kwargs)
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
