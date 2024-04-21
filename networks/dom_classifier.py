from torch import nn


class Classifier(nn.Module):
    def __init__(self, spatial_dims=2):
        super().__init__()

        if spatial_dims == 2:
            self.net = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.MaxPool2d(kernel_size=2),
                nn.LeakyReLU(0.01),
                nn.Conv2d(256, 1, kernel_size=3, padding=1),
                nn.MaxPool2d(kernel_size=2),
                nn.LeakyReLU(0.01),
                nn.AdaptiveAvgPool2d(output_size=1),
            )
        elif spatial_dims == 3:
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
        else:
            raise Exception("spatial dim: 2, 3")

    def forward(self, x):
        return self.net(x)
