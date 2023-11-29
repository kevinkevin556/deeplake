# Mixin classes can be inherited to modify components of modules or
# alter the update process of neural network during training.
#
# Inheriting from a mixin class should be an exclusive method to
# make changes to the pre-built modules in this repository.


import torch
import torch.nn.functional as F
from einops import einsum
from monai.losses import DiceCELoss
from torch import nn

from lib.loss.target_adaptive_loss import TargetAdaptiveLoss


class BaseMixin(nn.Module):
    pass


class PredictDomainEquivalenceMixin(BaseMixin):
    """
    Description: Instead of predicting the modalities of the pair of images in each round,
                 predict whether two images drawn from the identical domain or not.

    Performance: Good.
    """

    def __init__(self):
        super().__init__()
        self.data_loading_mode = "random"
        self.dom_classifier = nn.Sequential(
            nn.Conv3d(512, 128, kernel_size=3, padding=1),
            nn.MaxPool3d(kernel_size=2),
            nn.ReLU(0.01),
            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            nn.MaxPool3d(kernel_size=2),
            nn.ReLU(0.01),
            nn.Flatten(),
            nn.Linear(64, 1),
        )

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


class ConterfactucalAlignmentMixin(BaseMixin):
    """
    Description: Align CAM and its counterfactual counterpart.

    Performance: Not an effective method for partially-supervised segmentation task
                 under domain shift.
    """

    def __init__(self):
        super().__init__()
        self.data_loading_mode = "random"
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

    def update(self, images, masks, modalities, alpha):
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


class RandomPairInputMixin(BaseMixin):
    """
    Description: Each time 2 random images (whose modality are not specified)
                 are loaded to update network's parameters.

    Performance:  Not effective.
    """

    def __init__(self):
        super().__init__()
        self.data_loading_mode = "random"
        self.dom_classifier = nn.Sequential(
            nn.Conv3d(256, 128, kernel_size=3, padding=1),
            nn.MaxPool3d(kernel_size=2),
            nn.ReLU(),
            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            nn.MaxPool3d(kernel_size=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64, 1),
        )

    def update(self, images, masks, modalities, alpha):
        self.grl.set_alpha(alpha)
        self.optimizer.zero_grad()

        _seg_losses = {}
        _adv_losses = {}
        for i in [0, 1]:
            # Predictor branch
            skip_outputs, feature = self.feat_extractor(images[i])
            output = self.predictor((skip_outputs, feature))
            if modalities[i][0] == "ct":
                _seg_losses[i] = self.ct_tal(output, masks[i])
            elif modalities[i][0] == "mr":
                _seg_losses[i] = self.mr_tal(output, masks[i])
            else:
                raise ValueError(f"Invalid modality {modalities[i][0]}")

            # Domain Classifier branch
            dom_pred_logits = self.dom_classifier(self.grl.apply(feature))
            dom_true_label = torch.tensor([[m == "ct"] for m in modalities[i]], device="cuda") * 1.0
            _adv_losses[i] = self.adv_loss(dom_pred_logits, dom_true_label)

        seg_loss = _seg_losses[0] + _seg_losses[1]
        seg_loss.backward(retain_graph=True)
        adv_loss = _adv_losses[0] + _adv_losses[1]
        adv_loss.backward()

        self.optimizer.step()
        return seg_loss.item(), adv_loss.item()


class CAMPseudoLabel(BaseMixin):
    """
    Desciption: Use from the last convolution layer as the psuedo label.

    Performance: Unknown.
    """

    def __init__(self):
        super().__init__()
        self.data_loading_mode = "paired"
        self.dom_classifier = nn.Sequential(
            nn.Conv3d(256, 128, kernel_size=3, padding=1),
            nn.MaxPool3d(kernel_size=2),
            nn.ReLU(),
            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            nn.MaxPool3d(kernel_size=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64, 1),
        )
        self.dice2loss = DiceCELoss(to_onehot_y=True, softmax=True)

    def update(self, images, masks, modalities, alpha, beta=0.75, gamma=0.2):
        self.grl.set_alpha(alpha)
        cam_threshold = gamma

        masks = list(masks)
        self.optimizer.zero_grad()

        def get_cam_weight(feature):
            jacobs = torch.empty((2, self.num_classes, *feature.shape[1:]))  # dim = (2, class, channel, h, w, d)
            gap = torch.nn.AvgPool3d(kernel_size=96)
            for patch in [0, 1]:
                jacobs[patch] = torch.squeeze(torch.autograd.functional.jacobian(gap, feature[[patch]]))
            return torch.mean(jacobs, dim=(3, 4, 5))

        def get_cam(feature):
            weight = get_cam_weight(feature).to("cuda")
            cam = einsum(weight, feature, "n cls ch, n ch h w d -> n cls h w d")
            return F.relu(cam)

        # Predictor branch
        _features = {}
        _seg_losses = {}
        _dom_pred_logits = {}
        for i in [0, 1]:
            skip_outputs, _features[i] = self.feat_extractor(images[i])
            output = self.predictor((skip_outputs, _features[i]))
            cam = get_cam(output)[:, 1:]
            cam *= cam > cam_threshold
            masks[i] += torch.argmax(cam, dim=1, keepdim=True) * (masks[i] == 0)
            cam_pseudo_label = torch.argmax(cam, dim=1, keepdim=True)
            cam_foreground = set(torch.unique(cam_pseudo_label).cpu().numpy()) - {0}

            if modalities[i][0] == "ct":
                foreground = list(set(self.ct_foreground) | cam_foreground)
            elif modalities[i][0] == "mr":
                foreground = list(set(self.mr_foreground) | cam_foreground)
            else:
                raise ValueError(f"Invalid modality {modalities[i][0]}")

            tal = TargetAdaptiveLoss(self.num_classes, foreground)
            _seg_losses[i] = tal(output, masks[i])
            _seg_losses[i].backward(retain_graph=True)
        seg_loss = _seg_losses[0] + _seg_losses[1]

        # Domain Classifier branch: predict domain equivalent
        for i in [0, 1]:
            _dom_pred_logits[i] = self.dom_classifier(self.grl.apply(_features[i]))
        dom_pred_logits = torch.cat([_dom_pred_logits[0], _dom_pred_logits[1]])
        dom_true_label = torch.cat(
            (
                torch.ones(_dom_pred_logits[0].shape, device="cuda"),
                torch.zeros(_dom_pred_logits[1].shape, device="cuda"),
            )
        )
        adv_loss = self.adv_loss(dom_pred_logits, dom_true_label)
        adv_loss.backward()

        self.optimizer.step()
        return seg_loss.item(), adv_loss.item()
