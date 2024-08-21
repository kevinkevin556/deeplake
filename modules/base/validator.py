from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
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


class BaseValidator:
    """
    The base class of validators.
    It is intended to be a template for users to create their own validator class.
    """

    def __init__(
        self,
        metric: Callable | Metric,
        is_train: bool = False,
        device: Literal["cuda", "cpu"] = "cuda",
        unpack_item: Callable | Literal["monai", "pytorch"] = "pytorch",
        output_infer: bool = True,
    ):
        """
        Args:
            metric (monai.Metric): The metric used to evaluate the model's performance.
            is_train (bool): Flag indicating if the validator is for training. Defaults to False.
            device (Literal["cuda", "cpu"]): The device to use for validation. Defaults to 'cuda'.
        """
        self.metric = metric
        self.is_train = is_train
        self.device = device
        self.output_infer = output_infer

        if unpack_item == "pytorch":
            self.unpack_item = lambda batch: (batch[0].to(self.device), batch[1].to(self.device))
        elif unpack_item == "monai":
            self.unpack_item = lambda batch: (batch["image"].to(self.device), batch["label"].to(self.device))
        else:
            self.unpack_item = unpack_item

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
    ) -> dict:

        module.eval()
        module.to(self.device)
        val_metrics = []

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
                images, targets = self.unpack_item(batch)

                # Get inferred / forwarded results of module
                if getattr(module, "inference", False) and self.output_infer:
                    infer_out = module.inference(images)
                else:
                    infer_out = module.forward(images)

                # Compute validation metrics
                batch_metric = self.metric(infer_out, targets).item()
                val_metrics += [batch_metric]

                # Update progressbar
                info = {
                    "metric_name": self.metric.__class__.__name__,
                    "batch_metric": batch_metric,
                    "global_step": global_step,
                }
                desc = self.pbar_description.format(**info)
                pbar.set_description(desc)

        output = np.mean(val_metrics)
        return output


class SmatDatasetValidator(BaseValidator):

    def __init__(
        self,
        metric: Metric,
        is_train: bool = False,
        device: Literal["cuda", "cpu"] = "cuda",
        output_infer: bool = True,
    ):
        super().__init__(metric, is_train, device, unpack_item="monai", output_infer=output_infer)
        if is_train:
            self.pbar_description = "Validate ({global_step} Steps) (Partially-labelled:{val_on_partial}) ({metric_name}={batch_metric:2.5f})"
        else:
            self.pbar_description = "Validate (Partially-labelled:{val_on_partial}) ({metric_name}={batch_metric:2.5f})"

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

        module.eval()
        val_metrics = {"ct": [], "mr": []}
        metric_means = {"mean": None, "ct": None, "mr": None}

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
                images, masks = self.unpack_item(batch)
                modality_label = batch["modality"][0]
                num_classes = int(batch["num_classes"])
                background_classes = batch["background_classes"].numpy().flatten()

                assert modality_label in set(["ct", "mr"]), f"Unknown/Invalid modality {modality_label}"
                assert 0 in background_classes, "0 should be included in background_classes"

                # Get inferred / forwarded results of module
                if getattr(module, "inference", False) and self.output_infer:
                    infer_out = module.inference(images, modality=modality_label)
                else:
                    infer_out = module.forward(images)

                # Discretize the prediction and masks of ground truths
                samples = decollate_batch({"prediction": infer_out, "ground_truth": masks})
                preds: list = discretize_and_backgroundify_preds(samples, num_classes, background_classes)
                masks: list = discretize_and_backgroundify_masks(samples, num_classes, background_classes)

                # Compute validation metrics
                self.metric(y_pred=preds, y=masks)
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

        metric_means["mean"] = np.mean(val_metrics["ct"] + val_metrics["mr"])
        metric_means["ct"] = np.mean(val_metrics["ct"]) if len(val_metrics["ct"]) > 0 else np.nan
        metric_means["mr"] = np.mean(val_metrics["mr"]) if len(val_metrics["mr"]) > 0 else np.nan
        return metric_means
