import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from torch import nn, ones, zeros
from torch.nn import BCELoss
from torch.optim import SGD, Adam
from tqdm import tqdm

from lib.loss.target_adaptive_loss import TargetAdaptiveLoss
from networks.uxnet3d.network_backbone import UXNETDecoder, UXNETEncoder

device = torch.device("cuda")


class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return -grad_output


class AdversialLoss(nn.Module):
    def forward(self, ct_dom_pred, mr_dom_pred):
        # label: ct = 0, mr = 1
        bce = BCELoss()
        y_pred = torch.cat((ct_dom_pred, mr_dom_pred))
        y_true = torch.cat((ones(len(ct_dom_pred)), zeros(len(mr_dom_pred))))
        return bce(y_pred, y_true)


class DANNModule:
    def __init__(self, ct_foreground: Optional[list] = None, mr_foreground: Optional[list] = None):
        self.feat_extractor = UXNETEncoder(in_chans=1)
        self.predictor = UXNETDecoder(out_chans=16)

        h, w, d = 96, 96, 96  # feature size
        self.grl = GradientReversalLayer()
        self.dom_classifier = nn.Sequential(
            nn.Linear((h / 32) * (w / 32) * (d / 32) * 768, 1024),
            nn.Linear(1024, 1024),
            nn.Linear(1024, 1),
        )

        params = list(self.feat_extractor) + list(self.predictor) + list(self.dom_classifer)
        self.optimizer = Adam([p for p in params if p.requires_grad])
        self.ct_tal = (
            TargetAdaptiveLoss(foreground=ct_foreground)
            if ct_foreground
            else DiceCELoss(to_onehot_y=True, softmax=True)
        )
        self.mr_tal = (
            TargetAdaptiveLoss(foreground=mr_foreground)
            if mr_foreground
            else DiceCELoss(to_onehot_y=True, softmax=True)
        )
        self.adv_loss = AdversialLoss()

    def forward(self, x, branch=0):
        feature, skip_outputs = self.feat_extractor(x)
        if branch == 0:
            output = self.predictor((feature, skip_outputs))
            return output
        if branch == 1:
            domain_prediction = self.dom_classifier(self.grl(feature))
            return domain_prediction

    def update(self, ct_image, ct_mask, mr_image, mr_mask):
        self.optimizer.zero_grad()

        # Predictor branch
        ct_feature, ct_skip_outputs = self.feat_extractor(ct_image)
        ct_output = self.predictor((ct_feature, ct_skip_outputs))
        mr_feature, mr_skip_outputs = self.feat_extractor(mr_image)
        mr_output = self.predictor(mr_feature, mr_skip_outputs)

        ct_seg_loss = self.ct_tal(ct_output, ct_mask)
        mr_seg_loss = self.mr_tal(mr_output, mr_mask)
        seg_loss = ct_seg_loss + mr_seg_loss
        seg_loss.backward()

        # Domain Classifier branch
        ct_dom_pred = self.dom_classifier(self.grl(ct_feature))
        mr_dom_pred = self.dom_classifier(self.grl(mr_feature))
        adv_loss = self.adv_loss(ct_dom_pred, mr_dom_pred)
        adv_loss.backward()

        self.optimizer.step()
        return seg_loss.item(), adv_loss.item()

    def inference(self, x, roi_size=(96, 96, 96), sw_batch_size=2):
        # Using sliding windows
        self.eval()
        return sliding_window_inference(x, roi_size, sw_batch_size, self.forward)

    def save(self, checkpoint_dir):
        torch.save(self.feat_extractor.state_dict, os.path.join(checkpoint_dir, "feat_extractor_state.pth"))
        torch.save(self.predictor.state_dict, os.path.join(checkpoint_dir, "predictor_state.pth"))
        torch.save(self.dom_classifier.state_dict, os.path.join(checkpoint_dir, "dom_classifier_state.pth"))


class DANNTrainer:
    def __init__(self, max_iter=10):
        self.max_iter = max_iter

    def fit(self, module, ct_dataloader, mr_dataloader):
        for it in range(self.max_iter):
            ct_image, ct_mask = next(iter(ct_dataloader))
            mr_image, mr_mask = next(iter(mr_dataloader))
            module.update(ct_image, ct_mask, mr_image, mr_mask)


class DANNTrainer:
    def __init__(
        self,
        max_iter=40000,
        metric=DiceMetric(include_background=True, reduction="mean", get_not_nans=False),
        eval_step=500,
        checkpoint_dir="./default_ckpt/",
        num_class=16,
    ):
        self.max_iter = max_iter
        self.metric = metric
        self.eval_step = eval_step
        self.checkpoint_dir = checkpoint_dir
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        self.postprocess = {"x": AsDiscrete(argmax=True, to_onehot=num_class), "y": AsDiscrete(to_onehot=num_class)}

    def validation(self, module, dataloader, global_step=None):
        module.eval()
        val_metrics = []
        val_pbar = tqdm(dataloader, dynamic_ncols=True)
        metric_name = self.metric.__class__.__name__
        train_val_desc = "Validate ({} Steps) ({}={:2.5f})"
        simple_val_desc = "Validate ({}={:2.5f})"
        with torch.no_grad():
            for batch in val_pbar:
                # Infer, decollate data into list of samples, and proprocess both predictions and labels
                images, masks = batch["image"].to(device), batch["label"].to(device)
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

        for step in train_pbar:
            module.train()
            batch = next(iter(train_dataloader))
            images, masks = batch["image"].to(device), batch["label"].to(device)
            loss = module.update(images, masks)
            train_pbar.set_description(f"Training ({step} / {self.max_iter} Steps) (loss={loss:2.5f})")

            if ((step + 1) % self.eval_step == 0) or (step == self.max_iter - 1):
                val_metric = self.validation(module, val_dataloader, global_step=step)
                if val_metric > best_metric:
                    module.save(self.checkpoint_dir)
                    print(f"Model saved! Validation: (New) {val_metric:2.7f} > (Old) {best_metric:2.7f}")
                    best_metric = val_metric
                else:
                    print(f"No improvement. Validation: (New) {val_metric:2.7f} <= (Old) {best_metric:2.7f}")

    def show_training_info(self, module, train_dataloader, val_dataloader):
        print("--------")
        print("# of Training Samples:", len(train_dataloader))
        print("# of Validation Samples:", len(val_dataloader))
        print("Max iteration:", self.max_iter, f"steps (validates per {self.eval_step} steps)")
        print("Checkpoint directory:", self.checkpoint_dir)
        print("Module Encoder:", module.feat_extractor.__class__.__name__)
        print("       Decoder:", module.predictor.__class__.__name__)
        print("Optimizer:", module.optimizer.__class__.__name__, f"(lr = {module.lr})")
        print("Loss function:", {"ct": module.ct_tal__class__.__name__, "mr": module.ct_tal__class__.__name__})
        print("Evaluation metric:", self.metric.__class__.__name__)
        print("--------")
