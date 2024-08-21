from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
import torch
import tqdm
from monai.data import DataLoader, decollate_batch
from monai.metrics import Metric
from torch import nn
from tqdm.auto import tqdm

from lib.datasets.dnb import (
    discretize_and_backgroundify_masks,
    discretize_and_backgroundify_preds,
)
from lib.tensor_shape import tensor
from modules.base.validator import BaseValidator


class CategoricalValidator(BaseValidator):
    def __init__(
        self,
        metric: Metric,
        num_classes: int,
        is_train: bool = False,
        device: Literal["cuda", "cpu"] = "cuda",
        pred_logits: bool = True,
    ):
        super().__init__(metric, is_train, device, pred_logits)
        self.num_classes = num_classes

    def validation(
        self,
        module: nn.Module,
        dataloader: DataLoader | Sequence[DataLoader],
        global_step: int | None = None,
    ) -> dict:
        """
        Perform the validation process.

        Args:
            module (nn.Module): The model to be validated.
            dataloader (DataLoader | Sequence[DataLoader]): The dataloader(s) providing the validation data.
            global_step (int | None): The current global step, if applicable.

        Returns:
            dict: A dictionary containing the mean validation metrics for 'ct' and 'mr' modalities and their overall mean.
        """

        if not isinstance(dataloader, (list, tuple)):
            dataloader = [dataloader]
        else:
            dataloader = [dl for dl in dataloader if dl is not None]

        module.eval()
        metric_means = {c: {"mean": None, "ct": None, "mr": None} for c in range(self.num_classes)}
        for c in range(self.num_classes):
            val_metrics = {"ct": [], "mr": []}
            pbar = tqdm(
                itertools.chain(*dataloader), total=sum(len(dl) for dl in dataloader), dynamic_ncols=True, disable=True
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

                    # Get inferred / forwarded results of module
                    if getattr(module, "inference", False):
                        try:
                            infer_out = module.inference(images, modality=modality_label)
                        except TypeError:
                            infer_out = module.inference(images)
                    else:
                        infer_out = module.forward(images)

                    # Discretize the prediction and masks of ground truths
                    samples = decollate_batch({"prediction": infer_out, "ground_truth": masks})
                    preds: list[tensor["c w d"]] = discretize_and_backgroundify_preds(
                        samples, num_classes, background_classes
                    )
                    masks: list[tensor["c w d"]] = discretize_and_backgroundify_masks(
                        samples, num_classes, background_classes
                    )
                    class_binary_preds = [1 * (torch.argmax(p, dim=0) == c)[None, None, :] for p in preds]
                    class_binary_masks = [m[None, [c], :] for m in masks]

                    # Compute validation metrics, omit the calculation of masked catergories during training
                    if (not self.is_train) or (c not in set(background_classes) - {0}):
                        self.metric(y_pred=class_binary_preds, y=class_binary_masks)
                        batch_metric = self.metric.aggregate().item()
                        val_metrics[modality_label] += [batch_metric]
                        self.metric.reset()
                    else:
                        batch_metric = np.nan
                        val_metrics[modality_label] += [batch_metric]

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


class CategoricalMinValidator(CategoricalValidator):
    def __init__(
        self,
        metric: Metric,
        num_classes: int,
        is_train: bool = False,
        device: Literal["cuda", "cpu"] = "cuda",
    ):
        super().__init__(metric, num_classes, is_train, device)

    def validation(
        self,
        module: nn.Module,
        dataloader: DataLoader | Sequence[DataLoader],
        global_step: int | None = None,
    ) -> dict:
        metric_means = super().validation(module, dataloader, global_step)
        tqdm.write(repr(pd.DataFrame.from_dict(metric_means, orient="index")))

        output = {}
        for modality in ("mean", "ct", "mr"):
            modality_metrics = [
                metric_means[c][modality] for c in metric_means.keys() if not np.isnan(metric_means[c][modality])
            ]
            output[modality] = min(modality_metrics) if modality_metrics else np.nan
        return output
