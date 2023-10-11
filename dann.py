import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete, Compose
from monai.utils import set_determinism
from torch import nn, ones, zeros
from torch.nn import BCEWithLogitsLoss
from torch.optim import SGD, Adam, AdamW
from tqdm import tqdm

from dataset import AMOSDataset
from lib.loss.target_adaptive_loss import TargetAdaptiveLoss
from networks.uxnet3d.network_backbone import UXNETDecoder, UXNETEncoder
from transforms import AddBackgroundClass

device = torch.device("cuda")


class GradientReversalLayer(torch.autograd.Function):
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
        self, ct_foreground: Optional[list] = None, mr_foreground: Optional[list] = None, optimizer="AdamW", lr=0.0001
    ):
        super().__init__()

        # Network components
        h, w, d = 96, 96, 96  # feature size
        self.num_class = 16  # number of AMOS classes
        self.ct_foreground = ct_foreground
        self.ct_background = list(set(range(1, self.num_class)) - set(ct_foreground))
        self.mr_foreground = mr_foreground
        self.mr_background = list(set(range(1, self.num_class)) - set(mr_foreground))

        self.feat_extractor = UXNETEncoder(in_chans=1)
        self.predictor = UXNETDecoder(out_chans=16)
        self.grl = GradientReversalLayer()
        self.dom_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear((h // 16) * (w // 16) * (d // 16) * 768, 1024),
            nn.Linear(1024, 1024),
            nn.Linear(1024, 1),
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
            TargetAdaptiveLoss(num_class=self.num_class, foreground=ct_foreground)
            if ct_foreground
            else DiceCELoss(to_onehot_y=True, softmax=True)
        )
        self.mr_tal = (
            TargetAdaptiveLoss(num_class=self.num_class, foreground=mr_foreground)
            if mr_foreground
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

    def update(self, ct_image, ct_mask, mr_image, mr_mask):
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
        torch.save(self.feat_extractor.state_dict, os.path.join(checkpoint_dir, "feat_extractor_state.pth"))
        torch.save(self.predictor.state_dict, os.path.join(checkpoint_dir, "predictor_state.pth"))
        torch.save(self.dom_classifier.state_dict, os.path.join(checkpoint_dir, "dom_classifier_state.pth"))


class DANNTrainer:
    def __init__(
        self,
        max_iter=40000,
        metric=DiceMetric(include_background=True, reduction="mean", get_not_nans=False),
        eval_step=500,
        checkpoint_dir="./default_ckpt/",
    ):
        self.max_iter = max_iter
        self.metric = metric
        self.eval_step = eval_step
        self.checkpoint_dir = checkpoint_dir
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def validation(self, module, ct_dtl, mr_dtl, global_step=None):
        module.eval()
        val_metrics = []
        num_class = module.num_class
        ct_background = module.ct_background
        mr_background = module.mr_background
        val_pbar = tqdm(range(len(ct_dtl) + len(mr_dtl)), dynamic_ncols=True)
        metric_name = self.metric.__class__.__name__
        train_val_desc = "Validate ({} Steps) ({}={:2.5f})"
        simple_val_desc = "Validate ({}={:2.5f})"
        with torch.no_grad():
            for i in val_pbar:
                # Set dataloader and post-processing
                if i % 2 == 0:
                    batch = next(iter(ct_dtl))
                    postprocess = {
                        "x": Compose(AsDiscrete(argmax=True, to_onehot=num_class), AddBackgroundClass(ct_background)),
                        "y": Compose(AsDiscrete(to_onehot=num_class), AddBackgroundClass(ct_background)),
                    }
                else:
                    batch = next(iter(mr_dtl))
                    postprocess = {
                        "x": Compose(AsDiscrete(argmax=True, to_onehot=num_class), AddBackgroundClass(mr_background)),
                        "y": Compose(AsDiscrete(to_onehot=num_class), AddBackgroundClass(mr_background)),
                    }
                # Infer, decollate data into list of samples, and proprocess both predictions and labels
                images, masks = batch["image"].to(device), batch["label"].to(device)
                samples = decollate_batch({"prediction": module.inference(images), "ground_truth": masks})
                outputs = [postprocess["x"](sample["prediction"]) for sample in samples]
                masks = [postprocess["y"](sample["ground_truth"]) for sample in samples]
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

        for step in train_pbar:
            module.train()
            ct_batch = next(iter(ct_train_dtl))
            mr_batch = next(iter(mr_train_dtl))
            ct_image, ct_mask = ct_batch["image"].to(device), ct_batch["label"].to(device)
            mr_image, mr_mask = mr_batch["image"].to(device), mr_batch["label"].to(device)
            seg_loss, adv_loss = module.update(ct_image, ct_mask, mr_image, mr_mask)
            train_pbar.set_description(
                f"Training ({step} / {self.max_iter} Steps) (seg_loss={seg_loss:2.5f}, adv_loss={adv_loss:2.5f})"
            )
            if ((step + 1) % self.eval_step == 0) or (step == self.max_iter - 1):
                val_metric = self.validation(module, ct_val_dtl, mr_val_dtl, global_step=step)
                if val_metric > best_metric:
                    module.save(self.checkpoint_dir)
                    print(f"Model saved! Validation: (New) {val_metric:2.7f} > (Old) {best_metric:2.7f}")
                    best_metric = val_metric
                else:
                    print(f"No improvement. Validation: (New) {val_metric:2.7f} <= (Old) {best_metric:2.7f}")

    def show_training_info(self, module, ct_train_dtl, ct_val_dtl, mr_train_dtl, mr_val_dtl):
        print("--------")
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


# CLI tool

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segementation branch of DANN, using AMOS dataset.")
    ## Input data hyperparameters
    parser.add_argument("--root", type=str, default="", required=True, help="Root folder of all your images and labels")
    parser.add_argument("--output", type=str, default="", required=True, help="Output folder for the best model")
    parser.add_argument("--ct-fg", type=int, nargs="+", required=True, help="Annotated organ on CT modality")
    parser.add_argument("--mr-fg", type=int, nargs="+", required=True, help="Annotated organ on MR modality")

    ## Input model & training hyperparameters
    # parser.add_argument("--mode", type=str, default="train", help="Training or testing mode")
    parser.add_argument("--batch_size", type=int, default="1", help="Batch size for subject input")
    # parser.add_argument("--crop_sample", type=int, default="2", help="Number of cropped sub-volumes for each subject")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate for training")
    parser.add_argument("--optim", type=str, default="AdamW", help="Optimizer types: Adam / AdamW")
    parser.add_argument("--max_iter", type=int, default=40000, help="Maximum iteration steps for training")
    parser.add_argument("--eval_step", type=int, default=500, help="Per steps to perform validation")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--dev", action="store_true")

    ## Efficiency hyperparameters
    # parser.add_argument("--gpu", type=str, default="0", help="your GPU number")
    parser.add_argument("--cache_rate", type=float, default=0.1, help="Cache rate to cache your dataset into GPUs")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers")

    args = parser.parse_args()

    if args.deterministic:
        set_determinism(seed=0)
        print("[deterministic mode]")

    ct_train_ds = AMOSDataset(args.root, "ct", "train", 3, None, args.dev, args.cache_rate, args.num_workers)
    ct_train_dtl = DataLoader(ct_train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    ct_val_ds = AMOSDataset(args.root, "ct", "validation", 3, None, args.dev, args.cache_rate, args.num_workers)
    ct_val_dtl = DataLoader(ct_val_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    mr_train_ds = AMOSDataset(args.root, "mr", "train", 3, None, args.dev, args.cache_rate, args.num_workers)
    mr_train_dtl = DataLoader(mr_train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    mr_val_ds = AMOSDataset(args.root, "mr", "validation", 3, None, args.dev, args.cache_rate, args.num_workers)
    mr_val_dtl = DataLoader(mr_val_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    module = DANNModule(
        ct_foreground=[i for i in range(1, 16) if i % 2 == 0], mr_foreground=[i for i in range(1, 16) if i % 2 == 1]
    )
    module.to(device)
    trainer = DANNTrainer(max_iter=args.max_iter, eval_step=args.eval_step)
    trainer.train(module, ct_train_dtl, ct_val_dtl, mr_train_dtl, mr_val_dtl)
