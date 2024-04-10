from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import Literal

import numpy as np
import torch
import tqdm
from medaset.transforms import BackgroundifyClasses
from monai.data import DataLoader, decollate_batch
from monai.metrics import Metric
from monai.transforms import AsDiscrete, Compose
from torch import nn
from tqdm.auto import tqdm

from modules.base.validator import BaseValidator, get_output_and_mask


class CategoricalValidator(BaseValidator):
    def __init__(
        self,
        metric: Metric,
        num_classes: int,
        is_train: bool = False,
        device: Literal["cuda", "cpu"] = "cuda",
    ):
        super().__init__(metric, is_train, device)
        self.num_classes = num_classes

    def validation(
        self,
        module: nn.Module,
        dataloader: DataLoader | Sequence[DataLoader],
        global_step: int | None = None,
    ) -> dict:

        if not isinstance(dataloader, (list, tuple)):
            dataloader = [dataloader]
        else:
            dataloader = [dl for dl in dataloader if dl is not None]

        module.eval()
        metric_means = {c: {"mean": None, "ct": None, "mr": None} for c in range(self.num_classes)}
        for c in range(self.num_classes):
            val_metrics = {"ct": [], "mr": []}
            pbar = tqdm(
                itertools.chain(*dataloader),
                total=sum(len(dl) for dl in dataloader),
                dynamic_ncols=True,
                disable=True,
            )
            with torch.no_grad():
                for batch in pbar:
                    # Infer, decollate data into list of samples, and postprocess both predictions and labels
                    images, masks = batch["image"].to(self.device), batch["label"].to(self.device)
                    modality_label = batch["modality"][0]
                    num_classes = int(batch["num_classes"])
                    background_classes = batch["background_classes"].numpy().flatten()

                    assert modality_label in set(["ct", "mr"]), f"Unknown/Invalid modality {modality_label}"
                    assert 0 in background_classes, "0 should be included in background_classes"

                    infer_out = module.inference(images)
                    samples = decollate_batch({"prediction": infer_out, "ground_truth": masks})
                    outputs, masks = get_output_and_mask(samples, num_classes, background_classes)
                    outputs = [1 * (torch.argmax(out, dim=0) == c)[None, None, :] for out in outputs]
                    masks = [m[None, [c], :] for m in masks]

                    # Compute validation metrics
                    self.metric(y_pred=outputs, y=masks)
                    batch_metric = self.metric.aggregate().item()
                    val_metrics[modality_label] += [batch_metric]
                    self.metric.reset()

                    # Update progressbar
                    info = {
                        "val_on_partial": set(background_classes) > set([0]),
                        "metric_name": self.metric.__class__.__name__,
                        "batch_metric": batch_metric,
                        "global_step": global_step,
                    }
                    desc = self.pbar_description.format(**info)
                    pbar.set_description(desc)

            metric_means[c]["mean"] = np.mean(val_metrics["ct"] + val_metrics["mr"])
            metric_means[c]["ct"] = np.mean(val_metrics["ct"]) if len(val_metrics["ct"]) > 0 else np.nan
            metric_means[c]["mr"] = np.mean(val_metrics["mr"]) if len(val_metrics["mr"]) > 0 else np.nan

        return metric_means
