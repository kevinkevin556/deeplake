from pathlib import Path

import torch
from torch import nn


class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5), padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5)),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=(2, 2), stride=2),
            nn.Flatten(),
            nn.Linear(in_features=5 * 5 * 16, out_features=120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(in_features=84, out_features=10),
        )

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001)  # called by the updater
        self.criterion = nn.CrossEntropyLoss()  # called by the updater

    def forward(self, x):
        return self.net(x)

    def inference(self, x):
        # called by the validator
        preds = self.forward(x)
        pred_labels = torch.argmax(preds, dim=1)
        return pred_labels

    def save(self, checkpoint_dir):
        # called by the validator
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), str(Path(checkpoint_dir) / "net.pth"))

    def load(self, checkpoint_dir):
        # called by the validator
        self.net.load_state_dict(torch.load(str(Path(checkpoint_dir) / "net.pth")))


def batch_acc(pred_labels, targets):
    # called by the validator
    correct = (pred_labels == targets).sum().item()
    total = len(targets)
    return torch.tensor(correct / total)
