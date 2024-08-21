import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import ToPILImage, ToTensor

train_dataset = MNIST("/files/", train=True, download=True, transform=ToTensor())
test_dataset = MNIST("/files/", train=False, download=True, transform=ToTensor())

train_dataset, valid_dataset = random_split(train_dataset, [0.9, 0.1], generator=torch.Generator().manual_seed(42))

train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_dataloader = DataLoader(valid_dataset, batch_size=128, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True)
