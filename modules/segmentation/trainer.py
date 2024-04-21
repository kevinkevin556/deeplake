from __future__ import annotations

from typing import Literal, Union

import numpy as np
from monai.data import DataLoader as MonaiDataLoader
from monai.metrics import DiceMetric, Metric
from torch.utils.data import DataLoader as PyTorchDataLoader

from modules.base.trainer import BaseTrainer

DataLoader = Union[MonaiDataLoader, PyTorchDataLoader]


class SegmentationTrainer(BaseTrainer):
    alias = "SegTrainer"

    def __init__(
        self,
        max_iter: int = 10000,
        eval_step: int = 500,
        metric: Metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False),
        checkpoint_dir: str = "./checkpoints/",
        device: Literal["cuda", "cpu"] = "cuda",
        dev: bool = False,
    ):
        super().__init__(max_iter, eval_step, metric, checkpoint_dir, device, dev)

    def train(
        self,
        module,
        updater,
        *,
        train_dataloader: DataLoader | None = None,
        val_dataloader: DataLoader | None = None,
        ct_dataloader: tuple[DataLoader, DataLoader] | None = None,
        mr_dataloader: tuple[DataLoader, DataLoader] | None = None,
    ):
        valid_ct_data = ct_dataloader[0] and ct_dataloader[1]
        valid_mr_data = mr_dataloader[0] and mr_dataloader[1]

        if valid_ct_data and valid_mr_data:
            train_dataloader = chain_shuffle(ct_dataloader[0], mr_dataloader[0])
            val_dataloader = chain_shuffle(ct_dataloader[1], mr_dataloader[1])
        elif valid_ct_data:
            train_dataloader, val_dataloader = ct_dataloader[0], ct_dataloader[1]
        elif valid_mr_data:
            train_dataloader, val_dataloader = mr_dataloader[0], mr_dataloader[1]
        else:
            # train_dataloader = train_dataloader
            # val_dataloader = val_dataloader
            pass

        if train_dataloader is None:
            raise ValueError(
                "No dataloader specified for training."
                "Please provide a valid train_dataloader or both ct_dataloader and mr_dataloader."
            )
        if val_dataloader is None:
            raise ValueError(
                "No dataloader specified for validation."
                "Please provide a valid val_dataloader or both ct_dataloader and mr_dataloader."
            )

        super().train(
            module,
            updater,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
        )


class ChainShuffleIterator:
    def __init__(self, *dataloaders):
        self.dataloaders = dataloaders
        self.num_dataloaders = len(dataloaders)
        self.total_batches = [len(dl) for dl in self.dataloaders]

    def __len__(self):
        return sum(self.total_batches)

    def __iter__(self):
        self.iterator_list = [iter(dl) for dl in self.dataloaders]
        self.remaining_batches = np.array(self.total_batches.copy())
        return self

    def __next__(self):
        assert sum(self.remaining_batches) >= 0, "Sum of remaining batches should be greater than or equal to 0"

        if np.sum(self.remaining_batches) > 0:
            rng = np.random.default_rng()
            sample_probs = self.remaining_batches / np.sum(self.remaining_batches)
            dataloader_idx = rng.choice(range(self.num_dataloaders), size=1, p=sample_probs)[0]
            self.remaining_batches[dataloader_idx] -= 1
            return next(self.iterator_list[dataloader_idx])
        else:
            raise StopIteration


def chain_shuffle(*dataloaders):
    return ChainShuffleIterator(*dataloaders)
