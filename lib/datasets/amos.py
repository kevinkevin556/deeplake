from __future__ import annotations

import numpy as np
from medaset import amos

from .dataset_wrapper import Dataset


class _AmosDatasetWithBackgroundInfo(amos.AmosDataset):
    def __init__(
        self,
        *args,
        background_classes: list | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if background_classes:
            self.background_classes = background_classes
        else:
            self.background_classes = [0]

    def __getitem__(self, index):
        output = super().__getitem__(index)
        if isinstance(output, dict):
            output["num_classes"] = self.num_classes
            output["background_classes"] = np.array(self.background_classes)
        if isinstance(output, list):
            for item in output:
                item["num_classes"] = self.num_classes
                item["background_classes"] = np.array(self.background_classes)
        return output


class AmosCtDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.in_use:
            if self.num_classes != 16:
                raise ValueError(f"There are 16 classes in AMOS dataset. Got {self.num_classes}")

            self.train_dataset = _AmosDatasetWithBackgroundInfo(
                root_dir=self.root_dir,
                modality="ct",
                stage="train",
                transform=self.train_transform,
                mask_mapping={c: 0 for c in self.train_background_classes},
                cache_rate=self.cache_rate,
                num_workers=self.num_workers,
                dev=self.dev,
                background_classes=self.train_background_classes,
            )
            setattr(self.train_dataset, "batch_size", self.train_batch_size)
            self.train_dataset = self.train_dataset[: -int(len(self.train_dataset) * self.holdout_ratio)]

            self.val_dataset = _AmosDatasetWithBackgroundInfo(
                root_dir=self.root_dir,
                modality="ct",
                stage="train",
                transform=self.test_transform,
                mask_mapping={c: 0 for c in self.train_background_classes},
                cache_rate=self.cache_rate,
                num_workers=self.num_workers,
                dev=self.dev,
                background_classes=self.train_background_classes,
            )
            self.val_dataset = self.val_dataset[-int(len(self.val_dataset) * self.holdout_ratio) :]

            self.test_dataset = _AmosDatasetWithBackgroundInfo(
                root_dir=self.root_dir,
                modality="ct",
                stage="validation",
                transform=self.test_transform,
                mask_mapping={c: 0 for c in self.test_background_classes},
                cache_rate=self.cache_rate,
                num_workers=self.num_workers,
                dev=self.dev,
                background_classes=self.test_background_classes,
            )


class AmosMrDataset(Dataset):
    def __init__(self, *args, sequence: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        if self.in_use:
            if sequence is not None:
                raise ValueError(f"No sequence to determine in AMOS dataset. Got {sequence}")
            if self.num_classes != 16:
                raise ValueError(f"There are 16 classes in AMOS dataset. Got {self.num_classes}")

            self.train_dataset = _AmosDatasetWithBackgroundInfo(
                root_dir=self.root_dir,
                modality="mr",
                stage="train",
                transform=self.train_transform,
                mask_mapping={c: 0 for c in self.train_background_classes},
                cache_rate=self.cache_rate,
                num_workers=self.num_workers,
                dev=self.dev,
                background_classes=self.train_background_classes,
            )
            setattr(self.train_dataset, "batch_size", self.train_batch_size)
            self.train_dataset = self.train_dataset[: -int(len(self.train_dataset) * self.holdout_ratio)]

            self.val_dataset = _AmosDatasetWithBackgroundInfo(
                root_dir=self.root_dir,
                modality="mr",
                stage="train",
                transform=self.test_transform,
                mask_mapping={c: 0 for c in self.train_background_classes},
                cache_rate=self.cache_rate,
                num_workers=self.num_workers,
                dev=self.dev,
                background_classes=self.train_background_classes,
            )
            self.val_dataset = self.val_dataset[-int(len(self.val_dataset) * self.holdout_ratio) :]

            self.test_dataset = _AmosDatasetWithBackgroundInfo(
                root_dir=self.root_dir,
                modality="mr",
                stage="validation",
                transform=self.test_transform,
                mask_mapping={c: 0 for c in self.test_background_classes},
                cache_rate=self.cache_rate,
                num_workers=self.num_workers,
                dev=self.dev,
                background_classes=self.test_background_classes,
            )
