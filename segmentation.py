import argparse
import os
import pdb
from pathlib import Path

import numpy as np
import torch
import tqdm
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.handlers import from_engine
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.utils import set_determinism
from torch import nn
from torch.optim import SGD, Adam, AdamW
from tqdm import tqdm

from dataset import AMOSDataset
from networks.UXNet_3D.network_backbone import UXNETDecoder, UXNETEncoder

device = torch.device("cuda")


class SegmentationModule(nn.Module):
    def __init__(self, criterion=DiceCELoss(to_onehot_y=True), optimizer="AdamW", lr=0.0001):
        super().__init__()
        self.feat_extractor = UXNETEncoder(in_chans=1)
        self.predictor = UXNETDecoder(out_chans=16)
        self.criterion = criterion

        params = list(self.feat_extractor.parameters()) + list(self.predictor.parameters())
        differentiable_params = [p for p in params if p.requires_grad]
        self.lr = lr
        if optimizer == "AdamW":
            self.optimizer = AdamW(differentiable_params, lr=self.lr)
        if optimizer == "Adam":
            self.optimizer = Adam(differentiable_params, lr=self.lr)
        if optimizer == "SGD":
            self.optimizer = SGD(differentiable_params, lr=self.lr)

    def forward(self, x):
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

    def inference(self, x, roi_size=(96, 96, 96), batch_size=2):
        # Using sliding windows
        return sliding_window_inference(x, roi_size, batch_size, self)

    def save(self, checkpoint_dir):
        torch.save(self.feat_extractor.state_dict, os.path.join(checkpoint_dir, "feat_extractor_state.pth"))
        torch.save(self.predictor.state_dict, os.path.join(checkpoint_dir, "predictor_state.pth"))


class SegmentationTrainer:
    def __init__(
        self, max_iter=40000, metric=DiceMetric(), eval_step=500, checkpoint_dir="./default_ckpt/", num_class=16
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
        val_pbar = tqdm(dataloader)
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
        train_pbar = tqdm(range(self.max_iter))

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
        print("Loss function:", module.criterion.__class__.__name__)
        print("Evaluation metric:", self.metric.__class__.__name__)
        print("--------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segementation branch of DANN, using AMOS dataset.")
    ## Input data hyperparameters
    parser.add_argument("--root", type=str, default="", required=True, help="Root folder of all your images and labels")
    parser.add_argument("--output", type=str, default="", required=True, help="Output folder for the best model")
    parser.add_argument("--modality", type=str, default="ct", help="ct / mr / ct+mr")

    ## Input model & training hyperparameters
    # parser.add_argument("--mode", type=str, default="train", help="Training or testing mode")
    parser.add_argument("--batch_size", type=int, default="1", help="Batch size for subject input")
    # parser.add_argument("--crop_sample", type=int, default="2", help="Number of cropped sub-volumes for each subject")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate for training")
    parser.add_argument("--optim", type=str, default="AdamW", help="Optimizer types: Adam / AdamW")
    parser.add_argument("--max_iter", type=int, default=40000, help="Maximum iteration steps for training")
    parser.add_argument("--eval_step", type=int, default=500, help="Per steps to perform validation")
    parser.add_argument("--deterministic", action="store_true")

    ## Efficiency hyperparameters
    # parser.add_argument("--gpu", type=str, default="0", help="your GPU number")
    parser.add_argument("--cache_rate", type=float, default=0.1, help="Cache rate to cache your dataset into GPUs")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers")

    args = parser.parse_args()

    if args.deterministic:
        set_determinism(seed=0)
        print("[deterministic]")

    train_dataset = AMOSDataset(
        root_dir=args.root,
        modality=args.modality,
        stage="train",
        spatial_dim=3,
        cache_rate=args.cache_rate,
        num_workers=args.num_workers,
    )
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_dataset = AMOSDataset(
        root_dir=args.root,
        modality=args.modality,
        stage="validation",
        spatial_dim=3,
        cache_rate=args.cache_rate,
        num_workers=args.num_workers,
    )
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    module = SegmentationModule(optimizer=args.optim, lr=args.lr)
    module.to(device)
    trainer = SegmentationTrainer(max_iter=args.max_iter, eval_step=args.eval_step)
    trainer.train(module, train_dataloader, val_dataloader)
