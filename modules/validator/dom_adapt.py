from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
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
from modules.base.validator import BaseValidator


class DomValidator(BaseValidator):
    def __init__(
        self,
        metric: Metric,
        is_train: bool = False,
        device: Literal["cuda", "cpu"] = "cuda",
    ):
        self.metric = metric
        self.is_train = is_train
        self.device = device

        if is_train:
            self.pbar_description = "Validate ({global_step} Steps) ({metric_name}={batch_metric:2.5f})"
        else:
            self.pbar_description = "Validate ({metric_name}={batch_metric:2.5f})"

    def __call__(
        self,
        module: nn.Module,
        dataloader: DataLoader | Sequence[DataLoader],
        global_step: int | None = None,
    ) -> dict:
        return self.validation(module, dataloader, global_step)

    def validation(
        self,
        module: nn.Module,
        dataloader: DataLoader | Sequence[DataLoader],
        global_step: int | None = None,
    ) -> float:

        val_metrics = []

        if not isinstance(dataloader, (list, tuple)):
            dataloader = [dataloader]
        else:
            dataloader = [dl for dl in dataloader if dl is not None]

        if global_step is not None:
            data_iter = iter(dataloader[0])
            n_data = len(dataloader[0])
        else:
            data_iter = iter(dataloader[1])
            n_data = len(dataloader[1])

        pbar = tqdm(data_iter, total=n_data, dynamic_ncols=True)

        module.eval()
        with torch.no_grad():
            for batch in pbar:
                # Infer, decollate data into list of samples, and postprocess both predictions and labels
                images, masks = batch["image"].to(self.device), batch["label"].to(self.device)
                modality_label = batch["modality"][0]
                num_classes = int(batch["num_classes"])
                background_classes = batch["background_classes"].numpy().flatten()

                assert modality_label in {"ct", "mr"}, f"Unknown/Invalid modality {modality_label}"
                assert 0 in background_classes, "0 should be included in background_classes"

                infer_out = module.inference(images)
                samples = decollate_batch({"prediction": infer_out, "ground_truth": masks})
                preds: list = discretize_and_backgroundify_preds(samples, num_classes, background_classes)
                masks: list = discretize_and_backgroundify_masks(samples, num_classes, background_classes)

                # Compute validation metrics
                self.metric(y_pred=preds, y=masks)
                batch_metric = self.metric.aggregate().item()
                val_metrics += [batch_metric]
                self.metric.reset()

                # Update progress bar
                info = {
                    "metric_name": self.metric.__class__.__name__,
                    "batch_metric": batch_metric,
                    "global_step": global_step,
                }
                desc = self.pbar_description.format(**info)
                pbar.set_description(desc)

        return np.mean(val_metrics)
