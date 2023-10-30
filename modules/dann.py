import argparse
import itertools
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from monai.data import DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, Metric
from monai.transforms import AsDiscrete, Compose
from torch import nn, ones, zeros
from torch.nn import BCEWithLogitsLoss
from torch.optim import SGD, Adam, AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from lib.loss.target_adaptive_loss import TargetAdaptiveLoss
from lib.utils.validation import get_output_and_mask
from networks.uxnet3d.network_backbone import UXNETDecoder, UXNETEncoder


class GradientReversalLayer(torch.autograd.Function):
    def __init__(ctx, alpha):
        ctx.set_alpha(alpha)

    def set_alpha(ctx, alpha):
        ctx.alpha = alpha

    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return -grad_output


class AdversarialLoss(nn.Module):
    def __init__(self, cuda=True):
        super().__init__()
        self.cuda = cuda

    def forward(self, ct_dom_pred_logits, mr_dom_pred_logits):
        # label: ct = 0, mr = 1
        bce = BCEWithLogitsLoss()
        y_pred = torch.cat((ct_dom_pred_logits, mr_dom_pred_logits))
        y_true = torch.cat((ones(ct_dom_pred_logits.shape), zeros(mr_dom_pred_logits.shape)))
        if self.cuda:
            y_true = y_true.to("cuda")
        return bce(y_pred, y_true)


class DANNModule(nn.Module):
    def __init__(
        self,
        num_classes: int,
        ct_foreground: Optional[list] = None,
        mr_foreground: Optional[list] = None,
        optimizer: str = "AdamW",
        lr: float = 0.0001,
    ):
        super().__init__()

        # Network components
        h, w, d = 96, 96, 96  # feature size
        self.num_classes = num_classes  # number of AMOS classes
        self.ct_foreground = ct_foreground
        self.ct_background = list(set(range(1, self.num_classes)) - set(ct_foreground)) if ct_foreground else None
        self.mr_foreground = mr_foreground
        self.mr_background = list(set(range(1, self.num_classes)) - set(mr_foreground)) if mr_foreground else None

        self.feat_extractor = UXNETEncoder(in_chans=1)
        self.predictor = UXNETDecoder(out_chans=self.num_classes)
        self.grl = GradientReversalLayer(alpha=1)
        self.dom_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear((h // 16) * (w // 16) * (d // 16) * 768, 512),
            # nn.Linear(1024, 1024),
            nn.Linear(512, 1),
        )

        # Optimizer
        params = (
            list(self.feat_extractor.parameters())
            + list(self.predictor.parameters())
            + list(self.dom_classifier.parameters())
        )
        self.lr = lr
        if optimizer == "AdamW":
            self.optimizer = AdamW(params, lr=self.lr)
        if optimizer == "Adam":
            self.optimizer = Adam(params, lr=self.lr)
        if optimizer == "SGD":
            self.optimizer = SGD(params, lr=self.lr)

        # Losses
        self.ct_tal = (
            TargetAdaptiveLoss(num_classes=self.num_classes, foreground=ct_foreground)
            if self.ct_background is not None
            else DiceCELoss(to_onehot_y=True, softmax=True)
        )
        self.mr_tal = (
            TargetAdaptiveLoss(num_classes=self.num_classes, foreground=mr_foreground)
            if self.mr_background is not None
            else DiceCELoss(to_onehot_y=True, softmax=True)
        )
        self.adv_loss = AdversarialLoss()

    def forward(self, x, branch=0):
        skip_outputs, feature = self.feat_extractor(x)
        if branch == 0:
            output = self.predictor((skip_outputs, feature))
            return output
        if branch == 1:
            dom_pred_logits = self.dom_classifier(self.grl.apply(feature))
            return dom_pred_logits

    def update(self, ct_image, ct_mask, mr_image, mr_mask, alpha=1):
        self.grl.set_alpha(alpha)
        self.optimizer.zero_grad()

        # Predictor branch
        ct_skip_outputs, ct_feature = self.feat_extractor(ct_image)
        ct_output = self.predictor((ct_skip_outputs, ct_feature))
        mr_skip_outputs, mr_feature = self.feat_extractor(mr_image)
        mr_output = self.predictor((mr_skip_outputs, mr_feature))

        ct_seg_loss = self.ct_tal(ct_output, ct_mask)
        mr_seg_loss = self.mr_tal(mr_output, mr_mask)
        seg_loss = ct_seg_loss + mr_seg_loss
        seg_loss.backward(retain_graph=True)

        # Domain Classifier branch
        ct_dom_pred_logits = self.dom_classifier(self.grl.apply(ct_feature))
        mr_dom_pred_logits = self.dom_classifier(self.grl.apply(mr_feature))
        adv_loss = self.adv_loss(ct_dom_pred_logits, mr_dom_pred_logits)
        adv_loss.backward()

        self.optimizer.step()
        return seg_loss.item(), adv_loss.item()

    def inference(self, x, roi_size=(96, 96, 96), sw_batch_size=2):
        # Using sliding windows
        self.eval()
        return sliding_window_inference(x, roi_size, sw_batch_size, self.forward)

    def save(self, checkpoint_dir):
        torch.save(self.feat_extractor.state_dict(), os.path.join(checkpoint_dir, "feat_extractor_state.pth"))
        torch.save(self.predictor.state_dict(), os.path.join(checkpoint_dir, "predictor_state.pth"))
        torch.save(self.dom_classifier.state_dict(), os.path.join(checkpoint_dir, "dom_classifier_state.pth"))

    def load(self, checkpoint_dir):
        try:
            self.feat_extractor.load_state_dict(torch.load(os.path.join(checkpoint_dir, "feat_extractor_state.pth")))
            self.predictor.load_state_dict(torch.load(os.path.join(checkpoint_dir, "predictor_state.pth")))
            self.dom_classifier.load_state_dict(torch.load(os.path.join(checkpoint_dir, "dom_classifier_state.pth")))
        except Exception as e:
            raise e

    def print_info(self):
        print("Module Encoder:", self.feat_extractor.__class__.__name__)
        print("       Decoder:", self.predictor.__class__.__name__)
        print("Optimizer:", self.optimizer.__class__.__name__, f"(lr = {self.lr})")
        print("Loss function:", {"ct": self.ct_tal.__class__.__name__, "mr": self.mr_tal.__class__.__name__})


class DANNTrainer:
    def __init__(
        self,
        max_iter: int = 10000,
        metric: Metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False),
        eval_step: int = 100,
        checkpoint_dir: str = "./checkpoints/",
        device: str = "cuda",
        data_info: dict = None,
        partially_labelled: bool = False,
    ):
        self.max_iter = max_iter
        self.metric = metric
        self.eval_step = eval_step
        self.checkpoint_dir = checkpoint_dir
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        self.device = device
        self.data_info = data_info
        self.partially_labelled = partially_labelled

    def show_training_info(self, module, train_dataloader, val_dataloader):
        ct_train_dtl, mr_train_dtl = train_dataloader
        ct_val_dtl, mr_val_dtl = val_dataloader
        print("--------")
        print("Device:", self.device)  # device is a global variable (not an argument of cli)
        print("# of Training Samples:", {"ct": len(ct_train_dtl), "mr": len(mr_train_dtl)})
        print("# of Validation Samples:", {"ct": len(ct_val_dtl), "mr": len(mr_val_dtl)})
        print("Max iteration:", self.max_iter, f"steps (validates per {self.eval_step} steps)")
        print("Checkpoint directory:", self.checkpoint_dir)
        print("Evaluation metric:", self.metric.__class__.__name__)
        module.print_info()
        print("--------")

    def validation(self, module, dataloader, global_step=None):
        module.eval()
        data_iter = itertools.chain(*dataloader)
        val_metrics = []
        num_classes = module.num_classes
        background = {"ct": module.ct_background, "mr": module.mr_background}
        val_pbar = tqdm(data_iter, total=len(dataloader[0]) + len(dataloader[1]), dynamic_ncols=True)
        metric_name = self.metric.__class__.__name__
        train_val_desc = "Validate ({} Steps) ({}={:2.5f})"
        simple_val_desc = "Validate ({}={:2.5f})"
        with torch.no_grad():
            for batch in val_pbar:
                # Infer, decollate data into list of samples, and proprocess both predictions and labels
                images, masks = batch["image"].to(self.device), batch["label"].to(self.device)
                modality_label = batch["modality"]
                samples = decollate_batch({"prediction": module.inference(images), "ground_truth": masks})
                if self.partially_labelled and (global_step is not None):
                    outputs, masks = get_output_and_mask(samples, num_classes, background[modality_label])
                else:
                    outputs, masks = get_output_and_mask(samples, num_classes)
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
        ct_train_dtl, mr_train_dtl = train_dataloader
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
            ## We gradually increase the value of lambda of grl as the training proceeds.
            #    p = float(batch_idx + epoch_idx * len_dataloader) / (n_epoch * len_dataloader)
            #    grl_lambda = 2. / (1. + np.exp(-10 * p)) - 1
            p = float(step) / self.max_iter
            grl_lambda = 2.0 / (1.0 + np.exp(-10 * p)) - 1
            seg_loss, adv_loss = module.update(ct_image, ct_mask, mr_image, mr_mask, alpha=grl_lambda)
            writer.add_scalar(f"train/seg_loss", seg_loss, step)
            writer.add_scalar(f"train/adv_loss", adv_loss, step)
            train_pbar.set_description(
                f"Training ({step} / {self.max_iter} Steps) (seg_loss={seg_loss:2.5f}, adv_loss={adv_loss:2.5f})"
            )
            if ((step + 1) % self.eval_step == 0) or (step == self.max_iter - 1):
                val_metric = self.validation(module, val_dataloader, global_step=step)
                writer.add_scalar(f"train/{self.metric.__class__.__name__}", val_metric, step)
                if val_metric > best_metric:
                    module.save(self.checkpoint_dir)
                    tqdm.write(f"Model saved! Validation: (New) {val_metric:2.7f} > (Old) {best_metric:2.7f}")
                    best_metric = val_metric
                else:
                    tqdm.write(f"No improvement. Validation: (New) {val_metric:2.7f} <= (Old) {best_metric:2.7f}")


class DANNInitializer:
    @staticmethod
    def init_dataloaders(train_dataset, val_dataset, test_dataset, batch_size, dev):
        ct_train_dataloader = DataLoader(train_dataset[0], batch_size=batch_size, shuffle=~dev, pin_memory=True)
        ct_val_dataloader = DataLoader(val_dataset[0], batch_size=1, shuffle=False, pin_memory=True)
        mr_train_dataloader = DataLoader(train_dataset[1], batch_size=batch_size, shuffle=~dev, pin_memory=True)
        mr_val_dataloader = DataLoader(val_dataset[1], batch_size=1, shuffle=False, pin_memory=True)
        ct_test_dataloader = (
            DataLoader(test_dataset[0], batch_size=1, shuffle=False, pin_memory=True) if test_dataset else None
        )
        mr_test_dataloader = (
            DataLoader(test_dataset[1], batch_size=1, shuffle=False, pin_memory=True) if test_dataset else None
        )
        return (
            (ct_train_dataloader, mr_train_dataloader),
            (ct_val_dataloader, mr_val_dataloader),
            (ct_test_dataloader, mr_test_dataloader),
        )

    @staticmethod
    def init_module(loss, optim, lr, dataset, modality, masked, device):
        if loss != "tal" and not masked:
            module = DANNModule(None, None, optim, lr, dataset.num_classes)
        else:
            module = DANNModule(dataset["fg"]["ct"], dataset["fg"]["mr"], optim, lr, dataset.num_classes)
        return module

    @staticmethod
    def init_trainer(max_iter, eval_step, checkpoint_dir, device):
        trainer = DANNTrainer(
            max_iter=max_iter,
            eval_step=eval_step,
            checkpoint_dir=checkpoint_dir,
            device=device,
        )
        return trainer
