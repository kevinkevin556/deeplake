import argparse
import os

import numpy as np
import torch
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from torch import autocast, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dann import UNetDecoder, UNetEncoder
from dataset import AMOSDataset


class UNet(nn.Module):
    def __init__(self, num_class=16):
        super().__init__()
        self.encoder = UNetEncoder(1, 1024)
        self.decoder = UNetDecoder(1024, num_class)

    def forward(self, x):
        skips, feat = self.encoder(x)
        y = self.decoder((skips, feat))
        return y


def train():
    train_dataset = AMOSDataset(
        root_dir=r"C:\Users\User\Desktop\TSM_Project\data\amos22",
        modality="ct",
        stage="train",
        dev=True,
    )
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=False)

    max_epoch = 500
    model = UNet()
    if os.path.exists("./checkpoint.pth"):
        print("Checkpoint exists. Load model from checkpoint.")
        model.load_state_dict(torch.load("./checkpoint.pth"))
    model.cuda()

    # criterion = DiceCELoss(to_onehot_y=True)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0005)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer, max_lr=0.01, steps_per_epoch=len(train_dataloader), epochs=max_epoch
    # )

    for e in range(max_epoch):
        epoch_train_loss = []
        pbar = tqdm(train_dataloader, position=0, leave=True)
        pbar.set_description(f"Epoch [{e+1}/{max_epoch}]")

        for image, label in pbar:
            # label = label[:, None, :, :]
            # with autocast("cuda"):  # automatic mixed precision
            pred = model(image)
            loss = criterion(pred, label)
            epoch_train_loss.append(loss.item())
            pbar.set_postfix(loss=loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()

        print(f" Epoch = {e+1}, Avg. training loss = {np.mean(epoch_train_loss)}")
        torch.save(model.state_dict(), "./checkpoint.pth")


def test():
    val_dataset = AMOSDataset(
        root_dir=r"C:\Users\User\Desktop\TSM_Project\data\amos22",
        modality="ct",
        stage="validation",
        dev=True,
    )
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, drop_last=False)

    model = UNet().cuda()
    model.load_state_dict(torch.load("./checkpoint.pth"))

    dice_metric = DiceMetric(include_background=True, reduction="mean")
    for image, label in val_dataloader:
        pred = model(image)
        dice_metric(y_pred=pred, y=label)

    print("Val dice score:", dice_metric.aggregate().item())
    dice_metric.reset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data for training or validation.")
    parser.add_argument(
        "stage",
        choices=["train", "val"],
        help='Specify the stage: "train" or "val".',
        default="train",
    )
    args = parser.parse_args()

    if args.stage == "train":
        train()
    elif args.stage == "val":
        test()
