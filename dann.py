import torch
from monai.networks.blocks import Convolution
from torch import nn, ones, zeros
from torch.nn import (
    BCELoss,
    Conv2d,
    ConvTranspose2d,
    MaxPool2d,
    Module,
    ReLU,
    Sequential,
)
from torch.optim import SGD, Adam
from torchvision.transforms.functional import center_crop

from tal import TargetAdaptiveLoss


class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return -grad_output


class AdversialLoss(Module):
    def forward(self, ct_dom_pred, mr_dom_pred):
        bce = BCELoss()
        pred = torch.cat((ct_dom_pred, mr_dom_pred))
        y = torch.cat((ones(len(ct_dom_pred)), zeros(len(mr_dom_pred))))
        return bce(pred, y)


class Conv(Sequential):
    def __init__(self, in_chns, out_chns, spatial_dims=2):
        super().__init__()
        self.add_module(
            "conv_0",
            Convolution(
                spatial_dims,
                in_chns,
                out_chns,
                kernel_size=3,
                act="PRELU",
                padding=1,
                norm="BATCH",
                bias=False,
            ),
        )


class TwoConv(Sequential):
    def __init__(self, in_chns, out_chns, spatial_dims=2):
        super().__init__()
        self.add_module(
            "conv_0",
            Convolution(
                spatial_dims,
                in_chns,
                out_chns,
                kernel_size=3,
                act="PRELU",
                padding=1,
                norm="BATCH",
                bias=False,
            ),
        )
        self.add_module(
            "conv_1",
            Convolution(
                spatial_dims,
                out_chns,
                out_chns,
                kernel_size=3,
                act="PRELU",
                padding=1,
                norm="BATCH",
                bias=False,
            ),
        )


class UpCat(Module):
    def __init__(self, in_chns, out_chns):
        super().__init__()
        self.upconv = ConvTranspose2d(in_chns, out_chns // 2, kernel_size=2, stride=2)

    def forward(self, x):
        skip_in, bottom_in = x
        bottom_in = self.upconv(bottom_in)
        skip_in = center_crop(skip_in, bottom_in.shape[2:])
        out = torch.cat((skip_in, bottom_in), dim=1)
        return out


class UNetEncoder(Module):
    def __init__(self, in_chns=1, out_chns=1024):
        super().__init__()
        self.conv0 = TwoConv(in_chns, 64)
        self.conv1 = Sequential(MaxPool2d(kernel_size=2), TwoConv(64, 128))
        self.conv2 = Sequential(MaxPool2d(kernel_size=2), TwoConv(128, 256))
        self.conv3 = Sequential(MaxPool2d(kernel_size=2), TwoConv(256, 512))
        self.feat_conv = Sequential(MaxPool2d(kernel_size=2), TwoConv(512, out_chns))

    def forward(self, x):
        x0 = self.conv0(x)
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        feature = self.feat_conv(x3)
        skips = (x0, x1, x2, x3)
        return skips, feature


class UNetDecoder(Module):
    def __init__(self, in_chns=1024, out_chns=2):
        super().__init__()
        self.upconv3 = Sequential(UpCat(in_chns, 1024), TwoConv(1024, 512))
        self.upconv2 = Sequential(UpCat(512, 512), TwoConv(512, 256))
        self.upconv1 = Sequential(UpCat(256, 256), TwoConv(256, 128))
        self.upconv0 = Sequential(UpCat(128, 128), TwoConv(128, 64), Conv2d(64, out_chns, kernel_size=1))

    def forward(self, x):
        skips, feature = x
        x0, x1, x2, x3 = skips
        y3 = self.upconv3((x3, feature))
        y2 = self.upconv2((x2, y3))
        y1 = self.upconv1((x1, y2))
        y0 = self.upconv0((x0, y1))
        return y0


class DANNModule:
    def __init__(self, ct_foreground: list, mr_foreground: list):
        self.feat_extractor = UNetEncoder()  # TODO
        self.predictor = UNetDecoder()  # TODO
        self.dom_classifier = None  # TODO

        params = list(self.feat_extractor) + list(self.predictor) + list(self.dom_classifer)
        self.optimizer = Adam([p for p in params if p.requires_grad])

        self.ct_tal = TargetAdaptiveLoss(foreground=ct_foreground)
        self.mr_tal = TargetAdaptiveLoss(foreground=mr_foreground)
        self.adv_loss = AdversialLoss()

    def update(self, ct_image, ct_mask, mr_image, mr_mask):
        self.optimizer.zero_grad()
        grl = GradientReversalLayer()

        # Predictor branch
        ct_feature, ct_skip_outputs = self.feat_extractor(ct_image)
        ct_output = self.predictor(ct_feature, ct_skip_outputs)
        ct_seg_loss = self.ct_tal(ct_output, ct_mask)

        mr_feature, mr_skip_outputs = self.feat_extractor(mr_image)
        mr_output = self.predictor(mr_feature, mr_skip_outputs)
        mr_seg_loss = self.mr_tal(mr_output, mr_mask)

        seg_loss = ct_seg_loss + mr_seg_loss
        seg_loss.backward()

        # Domain Classifier branch
        ct_dom_pred = self.dom_classifier(grl(ct_feature))
        mr_dom_pred = self.dom_classifier(grl(mr_feature))
        adv_loss = self.adv_loss(ct_dom_pred, mr_dom_pred)
        adv_loss.backward()

        self.optimizer.step()


class DANNTrainer:
    def __init__(self, max_iter=10):
        self.max_iter = max_iter

    def fit(self, module, ct_dataloader, mr_dataloader):
        for it in range(self.max_iter):
            ct_image, ct_mask = next(iter(ct_dataloader))
            mr_image, mr_mask = next(iter(mr_dataloader))
            module.update(ct_image, ct_mask, mr_image, mr_mask)
