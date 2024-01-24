from torch import nn


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(256, 128, kernel_size=3, padding=1),
            nn.MaxPool3d(kernel_size=2),
            nn.ReLU(0.01),
            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            nn.MaxPool3d(kernel_size=2),
            nn.ReLU(0.01),
            nn.Flatten(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)
