import argparse
import json
import os
import pdb
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import tqdm
from medaset.amos import AMOSDataset, amos_train_transforms, amos_val_transforms
from monai.data import DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.utils import set_determinism
from torch import nn
from torch.optim import SGD, Adam, AdamW
from torch.utils.data import ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from lib.loss.target_adaptive_loss import TargetAdaptiveLoss
from networks.uxnet3d.network_backbone import UXNETDecoder, UXNETEncoder

device = torch.device("cuda")
crop_sample = 2


class SegmentationModule(nn.Module):
    def __init__(self, criterion=DiceCELoss(to_onehot_y=True, softmax=True), optimizer="AdamW", lr=0.0001):
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
        # y = self.net(x)
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


class SegmentationTrainer:
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
        self.checkpoint_dir = Path(checkpoint_dir) / datetime.now().strftime("%Y%m%d-%H%M%S")
        self.postprocess = {
            "x": AsDiscrete(argmax=True, to_onehot=AMOSDataset.num_classes),
            "y": AsDiscrete(to_onehot=AMOSDataset.num_classes),
        }

    # auxilary function to show training info before training procedure starts
    def show_training_info(self, module, train_dataloader, val_dataloader):
        print("--------")
        print("Device:", device)  # device is a global variable (not an argument of cli)
        print("# of Training Samples:", len(train_dataloader))
        print("# of Validation Samples:", len(val_dataloader))
        print("Max iteration:", self.max_iter, f"steps (validates per {self.eval_step} steps)")
        print("Checkpoint directory:", self.checkpoint_dir)
        print("Module Encoder:", module.feat_extractor.__class__.__name__)
        print("       Decoder:", module.predictor.__class__.__name__)
        print("Optimizer:", module.optimizer.__class__.__name__, f"(lr = {module.lr})")
        print("Loss function:", repr(module.criterion))
        print("Evaluation metric:", self.metric.__class__.__name__)
        print("--------")

    def validation(self, module, dataloader, global_step=None):
        module.eval()
        val_metrics = []
        val_pbar = tqdm(dataloader, dynamic_ncols=True)
        metric_name = self.metric.__class__.__name__
        train_val_desc = "Validate ({} Steps) ({}={:2.5f})"  # progress bar description used during training
        simple_val_desc = "Validate ({}={:2.5f})"  # progress bar description used when the network is tested
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
        writer = SummaryWriter(log_dir=self.checkpoint_dir)
        writer.add_scalar(f"train/{self.metric.__class__.__name__}", 0, 0)  # validation metric starts from zero
        for step in train_pbar:
            module.train()
            batch = next(iter(train_dataloader))
            # Backpropagation
            images, masks = batch["image"].to(device), batch["label"].to(device)
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


def split_train_data(modality: str, bg_mapping: dict, data_config: dict):
    _configs = data_config.copy()
    _configs["modality"] = modality
    ## Training set and validation set are generated by spliting the original training set.
    ## Testing set is the validation set from the original data.
    ## ** note: We test the network without masking any annotation of the given organs.
    ##          Thus mask_mapping is assigned None.
    train_dataset = AMOSDataset(stage="train", transform=amos_train_transforms, mask_mapping=bg_mapping, **_configs)
    val_dataset = AMOSDataset(stage="train", transform=amos_val_transforms, mask_mapping=bg_mapping, **_configs)
    test_dataset = AMOSDataset(stage="validation", transform=amos_val_transforms, mask_mapping=None, **_configs)
    # 10% of the original training data is used as validation set.
    train_dataset = train_dataset[: -int(len(train_dataset) * 0.1)]
    val_dataset = val_dataset[-int(len(val_dataset) * 0.1) :]
    return train_dataset, val_dataset, test_dataset


# CLI tool

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segementation branch of DANN, using AMOS dataset.")
    # Input data hyperparameters
    parser.add_argument("--root", type=str, default="", required=True, help="Root folder of all your images and labels")
    parser.add_argument("--output", type=str, default="checkpoints", help="Output folder for the best model")
    parser.add_argument("--modality", type=str, default="ct", help="Modality type: ct / mr / ct+mr")
    parser.add_argument("--masked", action="store_true", help="If true, train with annotation-masked data")
    # Input model & training hyperparameters
    parser.add_argument("--mode", type=str, default="train", help="Mode: train / test")
    parser.add_argument(
        "--pretrained",
        type=str,
        help="Checkpointing dir of pretrained network for testing or continuing training procedure.",
    )
    parser.add_argument(
        "--train_data",
        type=str,
        default="split",
        help="Training data: 'all' (training sample) / 'split' (into training and val sets)",
    )
    parser.add_argument("--batch_size", type=int, default="1", help="Batch size for subject input")
    parser.add_argument("--loss", type=str, default="dice2", help="Loss: dice2 (=DiceCE) / tal")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate for training")
    parser.add_argument("--optim", type=str, default="AdamW", help="Optimizer types: Adam / AdamW")
    parser.add_argument("--max_iter", type=int, default=40000, help="Maximum iteration steps for training")
    parser.add_argument("--eval_step", type=int, default=500, help="Per steps to perform validation")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Developer mode. Only a small fraction of data are loaded, and temp checkpoints are saved in the directory 'debug/'",
    )
    # Efficiency hyperparameters
    parser.add_argument("--cache_rate", type=float, default=0.1, help="Cache rate to cache your dataset into GPUs")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers")

    args = parser.parse_args()

    # Whether train without randomness
    if args.deterministic:
        set_determinism(seed=0)
        print("** Deterministic = True")

    # background and foreground for both modalities (if --masked is applied)
    bg = {
        "ct": {i: 0 for i in range(1, AMOSDataset.num_classes) if i % 2 == 0},
        "mr": {i: 0 for i in range(1, AMOSDataset.num_classes) if i % 2 == 1},
    }
    fg = {
        "ct": [i for i in range(1, AMOSDataset.num_classes) if i % 2 == 1],
        "mr": [i for i in range(1, AMOSDataset.num_classes) if i % 2 == 0],
    }
    print("** Modality =", args.modality)
    # Datasets
    print("** Training set =", args.train_data)
    data_config = {
        "root_dir": args.root,
        "modality": args.modality,
        "cache_rate": args.cache_rate,
        "num_workers": args.num_workers,
        "dev": args.dev,
    }
    if args.train_data == "all":
        if not args.masked:
            print("** Foreground =", list(range(1, AMOSDataset.num_classes)))
            train_dataset = AMOSDataset(stage="train", mask_mapping=None, **data_config)
            val_dataset = AMOSDataset(stage="validation", mask_mapping=None, **data_config)
            test_dataset = None
            assert train_dataset.modality == args.modality
            assert val_dataset.modality == args.modality
        else:
            # Annotation masked
            print("** Annotation masked = True")
            if args.modality in ["ct", "mr"]:
                print("** Foreground =", fg[args.modality])
                train_dataset = AMOSDataset(stage="train", mask_mapping=bg[args.modality], **data_config)
                val_dataset = AMOSDataset(stage="validation", mask_mapping=bg[args.modality], **data_config)
                test_dataset = None
                assert train_dataset.modality == args.modality
                assert val_dataset.modality == args.modality
            else:
                # --modality = ct+mr
                print("** Foreground =")
                print("   - ct:", fg["ct"])
                print("   - mr:", fg["mr"])
                # Read ct data
                data_config["modality"] = "ct"
                ct_train_dataset = AMOSDataset(stage="train", mask_mapping=bg["ct"], **data_config)
                ct_val_dataset = AMOSDataset(stage="validation", mask_mapping=bg["ct"], **data_config)
                # Read mr data
                data_config["modality"] = "mr"
                mr_train_dataset = AMOSDataset(stage="train", mask_mapping=bg["mr"], **data_config)
                mr_val_dataset = AMOSDataset(stage="validation", mask_mapping=bg["mr"], **data_config)
                # Combine ct and mr data into training and validation set
                train_dataset = ConcatDataset([ct_train_dataset, mr_train_dataset])
                val_dataset = ConcatDataset([ct_val_dataset, mr_val_dataset])
                test_dataset = None
    elif args.train_data == "split":
        if args.modality in ["ct", "mr"]:
            if args.masked:
                print("** Annotation masked = True")
                print("** Foreground =", fg[args.modality])
                bg_mapping = bg[args.modality]
            else:
                print("** Foreground =", list(range(1, AMOSDataset.num_classes)))
                bg_mapping = None
            train_dataset, val_dataset, test_dataset = split_train_data(args.modality, bg_mapping, data_config)
        else:
            if args.masked:
                print("** Annotation masked = True")
                print("** Foreground =")
                print("   - ct:", fg["ct"])
                print("   - mr:", fg["mr"])
                ct_bg_mapping, mr_bg_mapping = bg["ct"], bg["mr"]
            else:
                print("** Foreground =", list(range(1, AMOSDataset.num_classes)))
                ct_bg_mapping, mr_bg_mapping = None, None
            ct_train_dataset, ct_val_dataset, ct_test_dataset = split_train_data("ct", ct_bg_mapping, data_config)
            mr_train_dataset, mr_val_dataset, mr_test_dataset = split_train_data("mr", mr_bg_mapping, data_config)
            train_dataset = ConcatDataset([ct_train_dataset, mr_train_dataset])
            val_dataset = ConcatDataset([ct_val_dataset, mr_val_dataset])
            test_dataset = ConcatDataset([ct_test_dataset, mr_test_dataset])
    else:
        raise ValueError("Got an invalid input of option --train_data.")

    # Dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True) if test_dataset else None

    # Loss function
    if args.loss == "tal":
        if args.modality in ["ct", "mr"]:
            criterion = TargetAdaptiveLoss(
                num_class=AMOSDataset.num_classes, foreground=fg[args.modality], device=device
            )
        else:
            raise NotImplementedError("Target adaptive loss does not support ct+mr currently.")
    else:
        criterion = DiceCELoss(include_background=True, to_onehot_y=True, softmax=True)

    # Initialize module (load module if pretrained checkpoint is provided)
    module = SegmentationModule(optimizer=args.optim, lr=args.lr, criterion=criterion)
    if args.pretrained:
        print("** Pretrained checkpoint =", args.pretrained)
        module.load(args.pretrained)
    module.to(device)

    # Train or test
    checkpoint_dir = args.output if not args.dev else "debug"
    # ** note: temp checkpoints are saved in the "debug" directory
    #          to separate the result of experiments and temporary
    #          checkpoints generated in developer mode.
    trainer = SegmentationTrainer(
        max_iter=args.max_iter,
        eval_step=args.eval_step,
        checkpoint_dir=checkpoint_dir,
    )
    print("** Mode =", args.mode)
    if args.mode == "train":
        # Save command-line arguments
        Path(trainer.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(trainer.checkpoint_dir) / "args.json", "w") as f:
            json.dump(vars(args), f, indent=4)
        # Train
        trainer.train(module, train_dataloader, val_dataloader)
        # Test after training
        if test_dataloader:
            test_metric = trainer.validation(module, test_dataloader)
            print("** Test (Final):", test_metric)
    elif args.mode == "test":
        if test_dataloader:
            test_metric = trainer.validation(module, test_dataloader)
            print("** Test (Final):", test_metric)
    else:
        raise ValueError("Got an invalid input of option --mode.")
