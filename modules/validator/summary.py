from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import pandas as pd
from monai.data import DataLoader
from monai.metrics import Metric
from torch import nn

from modules.base.validator import BaseValidator
from modules.validator.categorical import CategoricalValidator


class SummmaryValidator(CategoricalValidator, BaseValidator):
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
        category_means = CategoricalValidator.validation(self, module, dataloader, global_step)
        category_table = pd.DataFrame.from_dict(category_means, orient="index")

        total_means = BaseValidator.validation(self, module, dataloader, global_step)
        total_table = pd.DataFrame({k: [v] for k, v in total_means.items()}, index=["all"])

        output = pd.concat([category_table, total_table])
        return output
