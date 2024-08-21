from __future__ import annotations

import itertools
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as F
import tqdm
from monai.data import DataLoader, decollate_batch
from torch import nn
from torchvision.utils import draw_segmentation_masks
from tqdm.auto import tqdm

from lib.datasets.dnb import (
    discretize_and_backgroundify_masks,
    discretize_and_backgroundify_preds,
)
from lib.tensor_shape import tensor
from modules.base.validator import BaseValidator

DEFAULT_COLORS = [
    "red",
    "blue",
    "yellow",
    "magenta",
    "green",
    "indigo",
    "darkorange",
    "cyan",
    "pink",
    "yellowgreen",
]


class SegVisualizer(BaseValidator):
    def __init__(
        self, num_classes: int, device: Literal["cuda", "cpu"] = "cuda", output_dir: str = "./images", ground_truth=True
    ):
        super().__init__(
            metric=None,
            is_train=False,
            device=device,
        )
        self.num_classes = (num_classes,)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.ground_truth = ground_truth

    def validation(
        self,
        module: nn.Module,
        dataloader: DataLoader | Sequence[DataLoader],
        global_step: int | None = None,
    ) -> dict:
        module.eval()

        if not isinstance(dataloader, (list, tuple)):
            dataloader = [dataloader]
        else:
            dataloader = [dl for dl in dataloader if dl is not None]
        data_iter = itertools.chain(*dataloader)
        pbar = tqdm(
            data_iter,
            total=sum(len(dl) for dl in dataloader),
            dynamic_ncols=True,
        )

        with torch.no_grad():
            for batch in pbar:
                # Infer, decollate data into list of samples, and postprocess both predictions and labels
                image: tensor["1 1 h w"] = torch.Tensor(batch["image"]).to(self.device)
                mask: tensor["1 1 h w"] = torch.Tensor(batch["label"]).to(self.device)
                modality_label = batch["modality"][0]
                num_classes = int(batch["num_classes"])
                background_classes = batch["background_classes"].numpy().flatten()

                assert modality_label in set(["ct", "mr"]), f"Unknown/Invalid modality {modality_label}"
                assert 0 in background_classes, "0 should be included in background_classes"

                # Get inferred / forwarded results of module
                if getattr(module, "inference", False):
                    try:
                        infer_out = module.inference(image, modality=modality_label)
                    except TypeError:
                        infer_out = module.inference(image)
                else:
                    infer_out = module.forward(image)

                # Discretize the prediction and mask of ground truths
                samples = decollate_batch({"prediction": infer_out, "ground_truth": mask})
                pred: list = discretize_and_backgroundify_preds(samples, num_classes, background=[0])
                mask: list = discretize_and_backgroundify_masks(samples, num_classes, background=[0])
                pred, mask = pred[0][1:, :], mask[0][1:, :]  # background is not plotted

                # Transform the single-channel image into a 3-channel (rgb) image
                image: tensor["1 1 h w"] = (image * 256).to(torch.uint8)
                image_rgb: tensor["3 h w"] = torch.concat([image, image, image], dim=1).squeeze()
                mask_overlayed_image = image_rgb
                if self.ground_truth:
                    mask_overlayed_image = draw_segmentation_masks(
                        image_rgb, mask.bool(), alpha=0.3, colors=DEFAULT_COLORS[:num_classes]
                    )
                pred_overlayed_image = draw_segmentation_masks(
                    image_rgb, pred.bool(), alpha=0.3, colors=DEFAULT_COLORS[:num_classes]
                )
                save(
                    imgs=[mask_overlayed_image, pred_overlayed_image],
                    name=self.output_dir / Path(batch["image"].meta["filename_or_obj"][0]).name,
                )


def save(imgs, name):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    fig.savefig(f"{str(name)}.png", bbox_inches="tight")
    plt.close(fig)
