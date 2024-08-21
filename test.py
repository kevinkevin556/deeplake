from __future__ import annotations

from pathlib import Path

import torch
from jsonargparse import CLI
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from torch import nn

from lib.datasets.dataset_wrapper import Dataset
from modules.base.validator import SmatDatasetValidator
from modules.validator.seg_visualizer import SegVisualizer


def setup(
    ct_data: Dataset,
    mr_data: Dataset,
    module: nn.Module,
    pretrained: str | None = None,
    evaluator: str = None,
    device: str = "cuda",
):
    module = module.to(device)

    # Load pretrained module / network
    if pretrained is not None:
        if getattr(module, "load", False):
            module.load(pretrained)
        else:
            module = module.to(device)
            module.load_state_dict(torch.load(pretrained))

    # default evaluator for testing: SummaryValidator with DiceMetric

    return ct_data, mr_data, module, evaluator, pretrained


def main():
    ct_data, mr_data, module, evaluator, pretrained = CLI(setup, parser_mode="omegaconf")
    ct_dataloader = ct_data.get_data()
    mr_dataloader = mr_data.get_data()

    if Path(pretrained).is_dir():
        pass
    else:
        pretrained = str(Path(pretrained).parents[0])

    if evaluator is None or evaluator == "summary" or evaluator == "all":
        num_classes = getattr(ct_data, "num_classes", getattr(mr_data, "num_classes", None))
        dice_evaluator = SmatDatasetValidator(
            metric=DiceMetric(include_background=True, reduction="mean", get_not_nans=False),
            num_classes=num_classes,
        )
        dice = dice_evaluator.validation(module, dataloader=(ct_dataloader[2], mr_dataloader[2]))
        print(dice)
        dice.to_csv(f"{pretrained}/dice.csv")

    if evaluator == "hausdorff" or evaluator == "all":
        num_classes = getattr(ct_data, "num_classes", getattr(mr_data, "num_classes", None))
        hausdorff_evaluator = SmatDatasetValidator(
            metric=HausdorffDistanceMetric(include_background=True, reduction="mean", get_not_nans=False),
            num_classes=num_classes,
        )
        hausdorff = hausdorff_evaluator.validation(module, dataloader=(ct_dataloader[2], mr_dataloader[2]))
        print(hausdorff)
        hausdorff.to_csv(f"{pretrained}/hausdorff.csv")

    if evaluator == "draw" or evaluator == "all":
        num_classes = getattr(ct_data, "num_classes", getattr(mr_data, "num_classes", None))
        evaluator = SegVisualizer(
            num_classes=num_classes,
            output_dir=f"{pretrained}/images/",
            ground_truth=False,
        )
        performance = evaluator.validation(module, dataloader=(ct_dataloader[2], mr_dataloader[2]))


if __name__ == "__main__":
    main()
    # CLI(main, parser_mode="omegaconf", formatter_class=RichHelpFormatter)
