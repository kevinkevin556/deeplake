from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import torch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss
from torch.optim import SGD, Adam, AdamW

from lib.cyclegan.utils import load_cyclegan
from lib.loss.info_nce import InfoNCE
from lib.loss.target_adaptative_loss import TargetAdaptativeLoss
from lib.misc import Concat
from lib.tensor_shape import tensor
from modules.base import BaseUpdater


class CycleGanContrasiveModule(nn.Module):
    def __init__(
        self,
        cyclegan_checkpoints_dir: str,
        net: nn.Module = None,
        roi_size: tuple = (512, 512),
        sw_batch_size: int = 1,
        ct_criterion: _Loss = TargetAdaptativeLoss(num_classes=4, background_classes=[0, 1, 2]),
        mr_criterion: _Loss = TargetAdaptativeLoss(num_classes=4, background_classes=[0, 3]),
        optimizer: str = "AdamW",
        lr: float = 0.0001,
        device: Literal["cuda", "cpu"] = "cuda",
        pretrained: Path | None = None,
    ):
        super().__init__()
        self.roi_size = roi_size
        self.sw_batch_size = sw_batch_size

        net = net.to(device)
        self.encoder = net.encoder  # feature extractor
        self.decoder = (
            Concat(net.decoder, net.segmentation_head) if getattr(net, "segmentation_head") else net.decoder
        )  # predictor

        self.cyclegan = load_cyclegan(cyclegan_checkpoints_dir, which_epoch="latest")
        self.dice2 = DiceCELoss(softmax=True, to_onehot_y=True)

        self.ct_criterion = ct_criterion
        self.mr_criterion = mr_criterion
        self.contrast_loss = InfoNCE(negative_mode="unpaired")

        # Set up the optimizer for training
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.lr = lr
        if optimizer == "AdamW":
            self.optimizer = AdamW(params, lr=self.lr)
        elif optimizer == "Adam":
            self.optimizer = Adam(params, lr=self.lr)
        elif optimizer == "SGD":
            self.optimizer = SGD(params, lr=self.lr, weight_decay=0.0002, momentum=0.9)
        else:
            raise ValueError("The specified optimizer is not current supported.")

        if pretrained:
            self.load(pretrained)

        # placeholders for prototypes
        self.ct_prototypes = {}
        self.ct_cluster_features = {}
        self.mr_prototypes = {}
        self.mr_cluster_features = {}

        # temperatures used when calculating NCE for each cluster
        self.ct_prototypes_t = {}
        self.mr_prototypes_t = {}

    def forward(self, x):
        encoded = self.encoder(x)
        output = self.decoder(*encoded)
        return output

    def train(self, mode=True):
        self.encoder.train(mode)
        self.decoder.train(mode)

    def inference(self, x):
        # Inference using the sliding window approach
        self.eval()
        return sliding_window_inference(x, self.roi_size, self.sw_batch_size, self.forward)

    def save(self, checkpoint_dir):
        torch.save(self.encoder.state_dict(), os.path.join(checkpoint_dir, "encoder_state.pth"))
        torch.save(self.decoder.state_dict(), os.path.join(checkpoint_dir, "decoder_state.pth"))

    def load(self, checkpoint_dir):
        try:
            self.encoder.load_state_dict(torch.load(os.path.join(checkpoint_dir, "encoder_state.pth")))
            self.decoder.load_state_dict(torch.load(os.path.join(checkpoint_dir, "decoder_state.pth")))
        except Exception as e:
            raise e

    # Display information about the encoder, decoder, optimizer, and losses
    def print_info(self):
        print("Module Encoder:", self.encoder.__class__.__name__)
        print("       Decoder:", self.decoder.__class__.__name__)
        print("Optimizer:", self.optimizer.__class__.__name__, f"(lr = {self.lr})")
        print("Segmentation Loss:", {"ct": self.ct_criterion, "mr": self.mr_criterion})
        print("Contrastive Loss:", self.contrast_loss)


class PartUpdaterCycleGanContrasive(BaseUpdater):
    def __init__(self):
        super().__init__()
        self.sampling_mode = "sequential"

    def check_module(self, module):
        # super().check_module(module)
        assert isinstance(
            module, CycleGanContrasiveModule
        ), "The specified module should inherit CycleGanContrasiveModule."
        for component in ("ct_criterion", "mr_criterion", "optimizer", "encoder", "decoder"):
            assert getattr(
                module, component, False
            ), f"The specified module should incoporate component/method: {component}"

    def update(self, module, images, masks, modalities):

        masks = list(masks)
        module.optimizer.zero_grad()

        ct, mr = "A", "B"
        fake_mr = module.cyclegan.generate_image(input_image=images[0], from_domain=ct)
        fake_ct = module.cyclegan.generate_image(input_image=images[1], from_domain=mr)

        for i in (0, 1):
            m = modalities[i][0]

            if m == "ct":
                ct_encoded = module.encoder(images[i])
                ct_output = module.decoder(*ct_encoded)
                ct_feature = ct_encoded[-1] if isinstance(ct_encoded, (list, tuple)) else ct_encoded
                ct_seg_loss = module.ct_criterion(ct_output, masks[i])

                if ct_seg_loss.isnan():
                    breakpoint()

                fake_mr.require_grad = False
                pseudo_softmax = module.decoder(*module.encoder(fake_mr))
                ct_pseudo_label = masks[i] + torch.argmax(pseudo_softmax, dim=1, keepdim=True) * (masks[i] == 0)
                ct_seg_loss += module.dice2(ct_output, ct_pseudo_label)

                if ct_seg_loss.isnan():
                    breakpoint()

            else:
                mr_encoded = module.encoder(images[i])
                mr_output = module.decoder(*mr_encoded)
                mr_feature = mr_encoded[-1] if isinstance(mr_encoded, (list, tuple)) else mr_encoded
                mr_seg_loss = module.mr_criterion(mr_output, masks[i])

                if mr_seg_loss.isnan():
                    breakpoint()

                fake_ct.require_grad = False
                pseudo_softmax = module.decoder(*module.encoder(fake_ct))
                mr_pseudo_label = masks[i] + torch.argmax(pseudo_softmax, dim=1, keepdim=True) * (masks[i] == 0)
                mr_seg_loss += module.dice2(mr_output, mr_pseudo_label)

                if mr_seg_loss.isnan():
                    breakpoint()

        seg_loss = ct_seg_loss + mr_seg_loss
        seg_loss.backward(retain_graph=True)

        # Prototypical Contrastive

        n, c, h, w = ct_feature.shape
        nhw = n * h * w
        num_classes = ct_output.shape[1]
        # alpha = 10
        momentum = 0.9

        for j in (0, 1):
            m = modalities[j][0]
            if m == "ct":
                resized_label: tensor[n, c, h, w] = F.interpolate(ct_pseudo_label, size=(h, w))
                feature: tensor[nhw, c] = ct_feature.transpose(1, -1).reshape(-1, c)
            else:
                resized_label: tensor[n, c, h, w] = F.interpolate(mr_pseudo_label, size=(h, w))
                feature: tensor[nhw, c] = mr_feature.transpose(1, -1).reshape(-1, c)

            label_flatten: tensor[nhw] = resized_label.transpose(1, -1).reshape(-1)

            for i in range(num_classes):
                # z = torch.sum(label_flatten == i)  # cluster size
                cluster_features = F.normalize(feature[label_flatten == i], dim=1)
                cluster_prototype = cluster_features.mean(dim=0)
                # cluster_t = (cluster_features - cluster_prototype).norm(dim=1).sum() / (z * np.log(z + alpha))
                if m == "ct":
                    module.ct_cluster_features[i] = cluster_features
                    module.ct_prototypes[i] = update_tensor(
                        module.ct_prototypes.get(i, None), cluster_prototype, momentum
                    )
                    # module.ct_prototypes_t[i] = cluster_t
                else:
                    module.mr_cluster_features[i] = cluster_features
                    module.mr_prototypes[i] = update_tensor(
                        module.mr_prototypes.get(i, None), cluster_prototype, momentum
                    )
                    # module.mr_prototypes_t[i] = cluster_t

        nce_loss = 0
        for k in range(num_classes):
            # Compute Contrastive loss for CT features
            query = module.ct_cluster_features[k]
            # If query is not empty
            if len(query) > 0:

                if no_nan_in(module.mr_prototypes[k]):
                    positive_key = module.mr_prototypes[k]
                else:
                    positive_key = module.ct_prototypes[k]

                negative_keys = []
                for j in range(num_classes):
                    if j != k:
                        if no_nan_in(module.ct_prototypes[j]):
                            negative_keys.append(module.ct_prototypes[j])
                        elif no_nan_in(module.mr_prototypes[j]):
                            negative_keys.append(module.mr_prototypes[j])
                        else:
                            # there is no prototype of this class (from both modality), skip to next category
                            pass
                negative_keys = torch.vstack(negative_keys).detach().clone()
                negative_keys.requires_grad = False

                nce_loss += module.contrast_loss(query, positive_key, negative_keys, temperature=0.1)

            # Compute Contrastive loss for MR features
            query = module.mr_cluster_features[k]
            # If query is not empty
            if len(query) > 0:

                if no_nan_in(module.ct_prototypes[k]):
                    positive_key = module.ct_prototypes[k]
                else:
                    positive_key = module.mr_prototypes[k]

                negative_keys = []
                for j in range(num_classes):
                    if j != k:
                        if no_nan_in(module.mr_prototypes[j]):
                            negative_keys.append(module.mr_prototypes[j])
                        elif no_nan_in(module.mr_prototypes[j]):
                            negative_keys.append(module.ct_prototypes[j])
                        else:
                            # there is no prototype of this class (from both modality), skip to next category
                            pass
                negative_keys = torch.vstack(negative_keys).detach().clone()
                negative_keys.requires_grad = False

                nce_loss += module.contrast_loss(query, positive_key, negative_keys, temperature=0.1)

        nce_loss.backward(retain_graph=True)
        module.optimizer.step()
        return seg_loss.item(), nce_loss.item()


def no_nan_in(x):
    return ~x.isnan().any()


def update_tensor(x, x_new, momentum):
    x_new = x_new.detach().clone()
    x_new.require_grad = False
    if x is None:
        return x_new

    x = x.detach().clone()
    x.require_grad = False
    if x.isnan().any():
        return x_new
    elif x_new.isnan().any():
        return x
    else:
        return x * momentum + x_new * (1 - momentum)
