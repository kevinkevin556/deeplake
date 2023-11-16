import itertools
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from einops import einsum
from monai.data import DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, Metric
from torch import nn, ones, zeros
from torch.nn import BCEWithLogitsLoss
from torch.nn import functional as F
from torch.optim import SGD, Adam, AdamW
from torch.utils.data import ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from lib.loss.target_adaptive_loss import TargetAdaptiveLoss
from lib.utils.validation import get_output_and_mask
from networks.unet.unet3d import BasicUNetDecoder, BasicUNetEncoder
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


class DANN2Module(nn.Module):
    def __init__(
        self,
        out_channels: int,
        num_classes: int,
        ct_foreground: Optional[list] = None,
        mr_foreground: Optional[list] = None,
        optimizer: str = "AdamW",
        lr: float = 0.0001,
    ):
        super().__init__()

        # Network components
        self.num_classes = num_classes  # number of classes of dataset
        self.ct_foreground = ct_foreground
        self.ct_background = list(set(range(1, self.num_classes)) - set(ct_foreground)) if ct_foreground else None
        self.mr_foreground = mr_foreground
        self.mr_background = list(set(range(1, self.num_classes)) - set(mr_foreground)) if mr_foreground else None

        self.feat_extractor = BasicUNetEncoder(in_channels=1)
        self.predictor = BasicUNetDecoder(out_channels=out_channels)
        self.grl = GradientReversalLayer(alpha=1)
        
        # domain classifier for update
        self.dom_classifier = nn.Sequential(
            nn.Conv3d(512, 128, kernel_size=3, padding=1),
            nn.MaxPool3d(kernel_size=2),
            nn.ReLU(),
            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            nn.MaxPool3d(kernel_size=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64, 1),
        )

        # domain classifier for update_v2
        self.dom_classifier = nn.Sequential(
            nn.Conv3d(18, 9, kernel_size=3, padding=1),
            nn.MaxPool3d(kernel_size=2),
            nn.ReLU(),
            nn.Conv3d(9, 9, kernel_size=3, padding=1),
            nn.MaxPool3d(kernel_size=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(9, 1),
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
        self.adv_loss = BCEWithLogitsLoss()

    def forward(self, x, branch=0):
        skip_outputs, feature = self.feat_extractor(x)
        if branch == 0:
            output = self.predictor((skip_outputs, feature))
            return output
        if branch == 1:
            dom_pred_logits = self.dom_classifier(self.grl.apply(feature))
            return dom_pred_logits

    def update(self, images, masks, modalities, alpha):
        self.grl.set_alpha(alpha)
        self.optimizer.zero_grad()

        # Predictor branch
        _features = {}
        _seg_losses = {}
        for i in [0, 1]:
            skip_outputs, _features[i] = self.feat_extractor(images[i])
            output = self.predictor((skip_outputs, _features[i]))
            if modalities[i][0] == "ct":
                _seg_losses[i] = self.ct_tal(output, masks[i])
            elif modalities[i][0] == "mr":
                _seg_losses[i] = self.mr_tal(output, masks[i])
            else:
                raise ValueError(f"Invalid modality {modalities[i][0]}")
        seg_loss = _seg_losses[0] + _seg_losses[1]
        seg_loss.backward(retain_graph=True)

        # Domain Classifier branch: predict domain equivalent
        mod_equiv = torch.tensor([[m0 == m1] for m0, m1 in zip(*modalities)], device="cuda") * 1.0
        features = torch.cat([_features[0], _features[1]], dim=1)
        pred_modal_equiv = self.dom_classifier(self.grl.apply(features))
        adv_loss = self.adv_loss(pred_modal_equiv, mod_equiv)
        adv_loss.backward()

        self.optimizer.step()
        return seg_loss.item(), adv_loss.item()

    def update_v2(self, images, masks, modalities, alpha):
        self.grl.set_alpha(alpha)
        self.optimizer.zero_grad()

        def get_skips(patch):
            return tuple([so[[patch]] for so in skip_outputs])

        def get_cam_weight(feature):
            jacobs = torch.empty((2, self.num_classes, *feature.shape[1:]))  # dim = (2, class, channel, h, w, d)
            gap = torch.nn.AvgPool3d(kernel_size=96)

            def func(p):
                return lambda x: torch.squeeze(gap(self.predictor((get_skips(p), x))))

            for patch in [0, 1]:
                jacobs[patch] = torch.squeeze(torch.autograd.functional.jacobian(func(patch), feature[[patch]]))
            return torch.mean(jacobs, dim=(3, 4, 5))

        def get_cam(feature, counterfactual=False):
            weight = get_cam_weight(feature).to("cuda")
            cam = einsum(weight, feature, "n cls ch, n ch h w d -> n cls h w d")
            if counterfactual:
                return -F.relu(cam)
            else:
                return F.relu(cam)

        # Predictor branch
        _features = {}
        _seg_losses = {}
        _cams = {}
        for i in [0, 1]:
            skip_outputs, _features[i] = self.feat_extractor(images[i])
            output = self.predictor((skip_outputs, _features[i]))
            _cams[i] = get_cam(_features[i], counterfactual=(i == 1 and modalities[0] != modalities[1]))

            if modalities[i][0] == "ct":
                _seg_losses[i] = self.ct_tal(output, masks[i])
            elif modalities[i][0] == "mr":
                _seg_losses[i] = self.mr_tal(output, masks[i])
            else:
                raise ValueError(f"Invalid modality {modalities[i][0]}")
            _seg_losses[i].backward(retain_graph=True)
        seg_loss = _seg_losses[0] + _seg_losses[1]

        # Domain Classifier branch: predict domain equivalent
        mod_equiv = torch.tensor([[m0 == m1] for m0, m1 in zip(*modalities)], device="cuda") * 1.0
        # features = torch.cat([_features[0], _features[1]], dim=1)
        cams = torch.cat([_cams[0], _cams[1]], dim=1)
        pred_modal_equiv = self.dom_classifier(self.grl.apply(cams))
        adv_loss = self.adv_loss(pred_modal_equiv, mod_equiv)
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
        print("Segmentation Loss:", {"ct": self.ct_tal, "mr": self.mr_tal})
        print("Discriminator Loss:", self.adv_loss)


class DANN2Trainer:
    def __init__(
        self,
        num_classes: int,
        max_iter: int = 10000,
        metric: Metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False),
        eval_step: int = 100,
        checkpoint_dir: str = "./checkpoints/",
        device: str = "cuda",
        data_info: dict = None,
        partially_labelled: bool = False,
    ):
        self.num_classes = num_classes
        self.max_iter = max_iter
        self.metric = metric
        self.eval_step = eval_step
        self.checkpoint_dir = checkpoint_dir
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        self.device = device
        self.data_info = data_info
        self.partially_labelled = partially_labelled

    def show_training_info(self, module, train_dataloader, val_dataloader):
        ct_train_data, mr_train_data = train_dataloader.dataset.datasets
        ct_val_data, mr_val_data = val_dataloader.dataset.datasets
        print("--------")
        print("Device:", self.device)  # device is a global variable (not an argument of cli)
        print("# of Training Samples:", {"ct": len(ct_train_data), "mr": len(mr_train_data)})
        print("# of Validation Samples:", {"ct": len(ct_val_data), "mr": len(mr_val_data)})
        print("Max iteration:", self.max_iter, f"steps (validates per {self.eval_step} steps)")
        print("Checkpoint directory:", self.checkpoint_dir)
        print("Evaluation metric:", self.metric.__class__.__name__)
        module.print_info()
        print("--------")

    def validation(self, module, dataloader, global_step=None):
        module.eval()
        val_metrics = []
        num_classes = module.num_classes
        background = {"ct": module.ct_background, "mr": module.mr_background}
        val_pbar = tqdm(dataloader, total=len(dataloader), dynamic_ncols=True)
        metric_name = self.metric.__class__.__name__
        train_val_desc = (
            "Validate ({} Steps) (Partially-labelled:{}) ({}={:2.5f})"  # progress bar description used during training
        )
        simple_val_desc = (
            "Validate (Partially-labelled:{}) ({}={:2.5f})"  # progress bar description used when the network is tested
        )
        val_on_partial = self.partially_labelled and (global_step is not None)
        with torch.no_grad():
            for batch in val_pbar:
                # Infer, decollate data into list of samples, and proprocess both predictions and labels
                images, masks = batch["image"].to(self.device), batch["label"].to(self.device)
                modality_label = batch["modality"][0]
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
                    val_pbar.set_description(
                        train_val_desc.format(global_step, val_on_partial, metric_name, batch_metric)
                    )
                else:
                    val_pbar.set_description(simple_val_desc.format(val_on_partial, metric_name, batch_metric))
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
            data_iter = iter(train_dataloader)
            batch1, batch2 = next(data_iter), next(data_iter)
            images = batch1["image"].to(self.device), batch2["image"].to(self.device)
            masks = batch1["label"].to(self.device), batch2["label"].to(self.device)
            mods = batch1["modality"], batch2["modality"]
            ## We gradually increase the value of lambda of grl as the training proceeds.
            #    p = float(batch_idx + epoch_idx * len_dataloader) / (n_epoch * len_dataloader)
            #    grl_lambda = 2. / (1. + np.exp(-10 * p)) - 1
            p = float(step) / self.max_iter
            grl_lambda = 2.0 / (1.0 + np.exp(-10 * p)) - 1
            seg_loss, adv_loss = module.update_v2(images, masks, mods, alpha=grl_lambda)
            writer.add_scalar(f"train/seg_loss", seg_loss, step)
            writer.add_scalar(f"train/adv_loss", adv_loss, step)
            train_pbar.set_description(
                f"Training ({step} / {self.max_iter} Steps) ({mods[0][0]},{mods[1][0]}) (seg_loss={seg_loss:2.5f}, adv_loss={adv_loss:2.5f})"
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


class DANN2Initializer:
    @staticmethod
    def init_dataloaders(train_dataset, val_dataset, test_dataset, batch_size, dev):
        train_dataloader = DataLoader(
            ConcatDataset(train_dataset), batch_size=batch_size, shuffle=~dev, pin_memory=True
        )
        val_dataloader = DataLoader(ConcatDataset(val_dataset), batch_size=1, shuffle=False, pin_memory=True)
        test_dataloader = DataLoader(ConcatDataset(test_dataset), batch_size=1, shuffle=False, pin_memory=True)
        return train_dataloader, val_dataloader, test_dataloader

    @staticmethod
    def init_module(out_channels, loss, optim, lr, data_info, modality, partially_labelled, device, **kwargs):
        if loss != "tal":
            ct_foreground = None
            mr_foreground = None
        else:
            ct_foreground = data_info["fg"]["ct"]
            mr_foreground = data_info["fg"]["mr"]

        module = DANN2Module(
            out_channels=out_channels,
            num_classes=data_info["num_classes"],
            ct_foreground=ct_foreground,
            mr_foreground=mr_foreground,
            optimizer=optim,
            lr=lr,
            **kwargs,
        )

        return module

    @staticmethod
    def init_trainer(num_classes, max_iter, eval_step, checkpoint_dir, device, data_info, partially_labelled, **kwargs):
        trainer = DANN2Trainer(
            num_classes=num_classes,
            max_iter=max_iter,
            eval_step=eval_step,
            checkpoint_dir=checkpoint_dir,
            device=device,
            data_info=data_info,
            partially_labelled=partially_labelled,
        )
        return trainer
