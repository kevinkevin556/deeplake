import itertools
from typing import Literal, Optional, Sequence, Union

import numpy as np
import torch
import tqdm
from medaset.transforms import BackgroundifyClasses
from monai.data import DataLoader, decollate_batch
from monai.metrics import Metric
from monai.transforms import AsDiscrete, Compose
from torch import nn
from tqdm.auto import tqdm


def get_output_and_mask(samples, num_classes, background=[0]):
    if isinstance(background, (np.ndarray, torch.Tensor)):
        background = background.tolist()
    if isinstance(background, tuple):
        background = list(background)
    assert isinstance(background, list)

    if background != [0]:
        postprocess = {
            "x": Compose(
                AsDiscrete(argmax=True, to_onehot=num_classes),
                BackgroundifyClasses(channel_dim=0, classes=background),
            ),
            "y": Compose(
                AsDiscrete(to_onehot=num_classes),
                BackgroundifyClasses(channel_dim=0, classes=background),
            ),
        }
    else:
        postprocess = {
            "x": AsDiscrete(argmax=True, to_onehot=num_classes),
            "y": AsDiscrete(to_onehot=num_classes),
        }

    outputs = [postprocess["x"](sample["prediction"]) for sample in samples]
    masks = [postprocess["y"](sample["ground_truth"]) for sample in samples]
    return outputs, masks


class BaseValidator:
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
            self.pbar_description = "Validate ({global_step} Steps) (Partially-labelled:{val_on_partial}) ({metric_name}={batch_metric:2.5f})"
        else:
            self.pbar_description = "Validate (Partially-labelled:{val_on_partial}) ({metric_name}={batch_metric:2.5f})"

    def __call__(
        self,
        module: nn.Module,
        dataloader: Union[DataLoader, Sequence[DataLoader]],
        global_step: Optional[int] = None,
    ) -> dict:
        return self.validation(module, dataloader, global_step)

    def validation(
        self,
        module: nn.Module,
        dataloader: Union[DataLoader, Sequence[DataLoader]],
        global_step: Optional[int] = None,
    ) -> dict:
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
            total=sum([len(dl) for dl in dataloader]),
            dynamic_ncols=True,
        )

        with torch.no_grad():
            for batch in pbar:
                # Infer, decollate data into list of samples, and postprocess both predictions and labels
                images, masks = batch["image"].to(self.device), batch["label"].to(self.device)
                modality_label = batch["modality"][0]
                num_classes = int(batch["num_classes"])
                background_classes = batch["background_classes"].numpy().flatten()

                assert modality_label in ("ct", "mr"), f"Unknown/Invalid modality {modality_label}"
                assert 0 in background_classes, f"0 should be included in background_classes"

                infer_out = module.inference(images)
                samples = decollate_batch({"prediction": infer_out, "ground_truth": masks})
                outputs, masks = get_output_and_mask(samples, num_classes, background_classes)

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

        metric_means["mean"] = np.mean(val_metrics["ct"] + val_metrics["mr"])
        metric_means["ct"] = np.mean(val_metrics["ct"]) if len(val_metrics["ct"]) > 0 else np.nan
        metric_means["mr"] = np.mean(val_metrics["mr"]) if len(val_metrics["mr"]) > 0 else np.nan
        return metric_means
