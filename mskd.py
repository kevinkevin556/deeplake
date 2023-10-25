import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import BasicUNet
from monai.networks.utils import one_hot
from monai.transforms import AsDiscrete, Compose
from torch import nn, ones, zeros
from torch.nn import BCEWithLogitsLoss
from torch.optim import SGD, Adam, AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from networks.uxnet3d.network_backbone import UXNET
from segmentation import SegmentationModule


class MSKDModule(nn.Module):
    def __init__(self, num_classes):
        super.__init__()
        self.num_classes = num_classes

        self.teacher0 = BasicUNet(in_channels=1, out_channels=num_classes)
        self.teacher1 = BasicUNet(in_channels=1, out_channels=num_classes)
        self.student = BasicUNet(in_channels=1, out_channels=num_classes)
        self.kl_loss = nn.KLDivLoss(reduction="None")
        self.optimizer = Adam(self.student.parameters(), lr=0.001)

    def forward(self, x):
        return self.student(x)

    def update(self, x):
        self.optimizer.zero_grad()
        bg0 = self.background0
        bg1 = self.background1
        lbd1 = self.lambda_background
        lbd2 = self.lambda_feature

        # Logit-wise supervision
        logit_stud = self.student(x)
        prob_stud = F.softmax(logit_stud)
        # teacher 0 & teacher 1
        logit0, logit1 = self.teacher0(x), self.teacher1(x)
        k0 = torch.argmax(logit0, dim=1, keepdim=True)
        k1 = torch.argmax(logit1, dim=1, keepdim=True)
        M0, M1 = (k0 > 0) * 1, (k1 > 0) * 1
        prob0, prob1 = F.softmax(logit0), F.softmax(logit1)
        prob0[:, bg0, :, :, :] = 0
        prob1[:, bg1, :, :, :] = 0
        prob0, prob1 = F.softmax(logit0), F.softmax(logit1)
        logit_loss0 = torch.mean(self.kl_loss(prob0, torch.sum(prob_stud * one_hot(k0), dim=1)) * M0)
        logit_loss1 = torch.mean(self.kl_loss(prob1, torch.sum(prob_stud * one_hot(k1), dim=1)) * M1)
        logit_loss = logit_loss0 + logit_loss1
        # background
        Mb = (1 - M0) * (1 - M1)
        prob_b = (prob0 + prob1) / 2
        logit_loss_b = torch.mean(
            self.kl_loss(prob_b, torch.sum(prob_stud * (1 - one_hot(k0)) * (1 - one_hot(k1)), dim=1)) * Mb
        )

        # Feature-wise supervision (l=1 in decoder)
        feat_stud = torch.sort(None, dim=1)  # TODO
        M0_layer = F.max_pool3d(M0, kernel_size=2)
        feat0 = torch.sort(None, dim=1)  # TODO
        feature_loss0 = torch.mean(self.kl_loss(feat0, feat_stud) * M0_layer)
        M1_layer = F.max_pool3d(M1, kernel_size=2)
        feat1 = torch.sort(None, dim=1)  # TODO
        feature_loss1 = torch.mean(self.kl_loss(feat1, feat_stud) * M1_layer)
        feature_loss = feature_loss0 + feature_loss1

        total_loss = logit_loss + lbd1 * logit_loss_b + lbd2 * feature_loss
        total_loss.backward()
        self.optimizer.step()
        return total_loss.item()

    def inference(self, x, roi_size=(96, 96, 96), sw_batch_size=2):
        # Using sliding windows
        self.eval()
        return sliding_window_inference(x, roi_size, sw_batch_size, self.forward)

    def save(self, checkpoint_dir):
        torch.save(self.teacher0.state_dict(), os.path.join(checkpoint_dir, "ct_teacher_state.pth"))
        torch.save(self.teacher1.state_dict(), os.path.join(checkpoint_dir, "mr_teacher_state.pth"))
        torch.save(self.student.state_dict(), os.path.join(checkpoint_dir, "student_state.pth"))

    def load(self, checkpoint_dir):
        try:
            self.feat_extractor.load_state_dict(torch.load(os.path.join(checkpoint_dir, "ct_teacher_state.pth")))
            self.predictor.load_state_dict(torch.load(os.path.join(checkpoint_dir, "predictor_state.pth")))
            self.student.load_state_dict(torch.load(os.path.join(checkpoint_dir, "student_state.pth")))
        except Exception as e:
            raise e


class SimpleTeacherTrainer:
    def __init__(
        self,
        max_iter=40000,
        metric=DiceMetric(include_background=True, reduction="mean", get_not_nans=False),
        eval_step=500,
        checkpoint_dir="./checkpoints/",
        device="cuda",
    ):
        self.max_iter = max_iter
        self.metric = metric
        self.eval_step = eval_step
        self.checkpoint_dir = checkpoint_dir
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        self.device = device

    def show_training_info(self, module, ct_train_dtl, ct_val_dtl, mr_train_dtl, mr_val_dtl):
        print("--------")
        print("Device:", self.device)  # device is a global variable (not an argument of cli)
        print("# of Training Samples:", {"ct": len(ct_train_dtl), "mr": len(mr_train_dtl)})
        print("# of Validation Samples:", {"ct": len(ct_val_dtl), "mr": len(mr_val_dtl)})
        print("Max iteration:", self.max_iter, f"steps (validates per {self.eval_step} steps)")
        print("Checkpoint directory:", self.checkpoint_dir)
        print("Module Encoder:", module.feat_extractor.__class__.__name__)
        print("       Decoder:", module.predictor.__class__.__name__)
        print("Optimizer:", module.optimizer.__class__.__name__, f"(lr = {module.lr})")
        print("Loss function:", {"ct": module.ct_tal.__class__.__name__, "mr": module.mr_tal.__class__.__name__})
        print("Evaluation metric:", self.metric.__class__.__name__)
        print("--------")

    def validation(self, module, ct_dtl, mr_dtl, global_step=None, **kwargs):
        module.eval()
        val_metrics = []
        num_classes = module.num_class
        val_pbar = tqdm(range(len(ct_dtl) + len(mr_dtl)), dynamic_ncols=True)
        metric_name = self.metric.__class__.__name__
        train_val_desc = "Validate ({} Steps) ({}={:2.5f})"
        simple_val_desc = "Validate ({}={:2.5f})"
        post = {
            "x": Compose(AsDiscrete(argmax=True, to_onehot=num_classes)),
            "y": Compose(AsDiscrete(to_onehot=num_classes)),
        }

        with torch.no_grad():
            for i in val_pbar:
                # Set dataloader and post-processing
                if i % 2 == 0:  # ct
                    batch = next(iter(ct_dtl))
                else:  # mr
                    batch = next(iter(mr_dtl))

                # Infer, decollate data into list of samples, and proprocess both predictions and labels
                images, masks = batch["image"].to(self.device), batch["label"].to(self.device)
                samples = decollate_batch({"prediction": module.inference(images), "ground_truth": masks})
                outputs = [post["x"](sample["prediction"]) for sample in samples]
                masks = [post["y"](sample["ground_truth"]) for sample in samples]
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

    def train(self, module, ct_train_dtl, ct_val_dtl, mr_train_dtl, mr_val_dtl):
        self.show_training_info(module, ct_train_dtl, ct_val_dtl, mr_train_dtl, mr_val_dtl)
        best_metric = 0
        train_pbar = tqdm(range(self.max_iter), dynamic_ncols=True)
        writer = SummaryWriter(log_dir=self.checkpoint_dir)
        writer.add_scalar(f"train/{self.metric.__class__.__name__}", 0, 0)  # validation metric starts from zero

        for step in train_pbar:
            module.train()
            ct_batch = next(iter(ct_train_dtl))
            mr_batch = next(iter(mr_train_dtl))
            ct_image, ct_mask = ct_batch["image"].to(self.device), ct_batch["label"].to(self.device)
            mr_image, mr_mask = mr_batch["image"].to(self.device), mr_batch["label"].to(self.device)
            loss = module.update(ct_image, ct_mask, mr_image, mr_mask)
            writer.add_scalar(f"train/seg_loss", loss, step)
            train_pbar.set_description(f"Training ({step} / {self.max_iter} Steps) (loss={loss:2.5f})")
            if ((step + 1) % self.eval_step == 0) or (step == self.max_iter - 1):
                val_metric = self.validation(module, ct_val_dtl, mr_val_dtl, global_step=step)
                writer.add_scalar(f"train/{self.metric.__class__.__name__}", val_metric, step)
                if val_metric > best_metric:
                    module.save(self.checkpoint_dir)
                    print(f"Model saved! Validation: (New) {val_metric:2.7f} > (Old) {best_metric:2.7f}")
                    best_metric = val_metric
                else:
                    print(f"No improvement. Validation: (New) {val_metric:2.7f} <= (Old) {best_metric:2.7f}")
                    print(f"No improvement. Validation: (New) {val_metric:2.7f} <= (Old) {best_metric:2.7f}")
