from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from monai.transforms import AsDiscrete, Compose

from medaset.transforms import BackgroundifyClasses


def discretize_and_backgroundify_preds(samples: Sequence, num_classes: int, background: Sequence = (0,)) -> list:
    """
    Argmax and one-hot-encoded the given prediction logits. Masked the background class if needed.

    Args:
        samples (Sequence): The input samples containing predictions and ground truth masks.
        num_classes (int): The number of classes for one-hot encoding.
        background (Sequence): The background classes to be considered during postprocessing. Defaults to (0,).

    Returns:
        List[torch.Tensor]: Processed predictions.
    """
    if isinstance(background, (np.ndarray, torch.Tensor)):
        background = background.tolist()
    if isinstance(background, tuple):
        background = list(background)
    assert isinstance(background, list)

    if background != [0]:
        postprocess_pred = Compose(
            AsDiscrete(argmax=True, to_onehot=num_classes),
            BackgroundifyClasses(channel_dim=0, classes=background),
        )
    else:
        postprocess_pred = AsDiscrete(argmax=True, to_onehot=num_classes)

    preds = [postprocess_pred(sample["prediction"]) for sample in samples]
    return preds


def discretize_and_backgroundify_masks(samples: Sequence, num_classes: int, background: Sequence = (0,)) -> list:
    """
    One-hot-encoded the given ground truth masks. Masked the background class if needed.

    Args:
        samples (Sequence): The input samples containing predictions and ground truth masks.
        num_classes (int): The number of classes for one-hot encoding.
        background (Sequence): The background classes to be considered during postprocessing. Defaults to (0,).

    Returns:
        List[torch.Tensor]: Processed masks.
    """
    if isinstance(background, (np.ndarray, torch.Tensor)):
        background = background.tolist()
    if isinstance(background, tuple):
        background = list(background)
    assert isinstance(background, list)

    if background != [0]:
        postprocess_mask = Compose(
            AsDiscrete(to_onehot=num_classes),
            BackgroundifyClasses(channel_dim=0, classes=background),
        )
    else:
        postprocess_mask = AsDiscrete(to_onehot=num_classes)

    masks = [postprocess_mask(sample["ground_truth"]) for sample in samples]
    return masks
