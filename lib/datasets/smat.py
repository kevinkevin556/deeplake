from __future__ import annotations

import numpy as np
from medaset import lake

from .dataset_wrapper import Dataset


class _SmatCtDatasetWithBackgroundInfo(lake.SmatCtDataset):
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


class _SmatMrDatasetWithBackgroundInfo(lake.SmatMrDataset):
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


class SmatCtDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.in_use:
            if self.num_classes != 4:
                raise ValueError(f"There are 4 classes in SMAT dataset. Got {self.num_classes}")

            self.train_dataset = _SmatCtDatasetWithBackgroundInfo(
                root_dir=self.root_dir,
                target="all",
                stage="train",
                transform=self.train_transform,
                mask_mapping={c: 0 for c in self.train_background_classes},
                cache_rate=self.cache_rate,
                num_workers=self.num_workers,
                dev=self.dev,
                random_seed=self.random_seed,
                split_ratio=(
                    (1 - self.holdout_ratio) ** 2,
                    (1 - self.holdout_ratio) * self.holdout_ratio,
                    self.holdout_ratio,
                ),
                sm_as_whole=True,
                background_classes=self.train_background_classes,
            )
            setattr(self.train_dataset, "batch_size", self.train_batch_size)

            self.val_dataset = _SmatCtDatasetWithBackgroundInfo(
                root_dir=self.root_dir,
                target="all",
                stage="validation",
                transform=self.test_transform,
                mask_mapping={c: 0 for c in self.train_background_classes},
                cache_rate=self.cache_rate,
                num_workers=self.num_workers,
                dev=self.dev,
                random_seed=self.random_seed,
                split_ratio=(
                    (1 - self.holdout_ratio) ** 2,
                    (1 - self.holdout_ratio) * self.holdout_ratio,
                    self.holdout_ratio,
                ),
                sm_as_whole=True,
                background_classes=self.train_background_classes,
            )

            self.test_dataset = _SmatCtDatasetWithBackgroundInfo(
                root_dir=self.root_dir,
                stage="test",
                target="all",
                transform=self.test_transform,
                mask_mapping={c: 0 for c in self.test_background_classes},
                cache_rate=self.cache_rate,
                num_workers=self.num_workers,
                dev=self.dev,
                random_seed=self.random_seed,
                split_ratio=(
                    (1 - self.holdout_ratio) ** 2,
                    (1 - self.holdout_ratio) * self.holdout_ratio,
                    self.holdout_ratio,
                ),
                sm_as_whole=True,
                background_classes=self.test_background_classes,
            )


class SmatMrDataset(Dataset):
    def __init__(self, *args, sequence: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        if self.in_use:
            if sequence is None:
                raise ValueError("The sequence should be specified when the dataset is to be loaded.")
            if self.num_classes != 4:
                raise ValueError(f"There are 4 classes in SMAT dataset. Got {self.num_classes}")

            self.train_dataset = _SmatMrDatasetWithBackgroundInfo(
                root_dir=self.root_dir,
                sequence=sequence,
                target="all",
                stage="train",
                transform=self.train_transform,
                mask_mapping={c: 0 for c in self.train_background_classes},
                cache_rate=self.cache_rate,
                num_workers=self.num_workers,
                dev=self.dev,
                random_seed=self.random_seed,
                split_ratio=(
                    (1 - self.holdout_ratio) ** 2,
                    (1 - self.holdout_ratio) * self.holdout_ratio,
                    self.holdout_ratio,
                ),
                pkd_only=False,
                non_pkd_only=True,
                background_classes=self.train_background_classes,
            )
            setattr(self.train_dataset, "batch_size", self.train_batch_size)

            self.val_dataset = _SmatMrDatasetWithBackgroundInfo(
                root_dir=self.root_dir,
                sequence=sequence,
                target="all",
                stage="validation",
                transform=self.test_transform,
                mask_mapping={c: 0 for c in self.train_background_classes},
                cache_rate=self.cache_rate,
                num_workers=self.num_workers,
                dev=self.dev,
                random_seed=self.random_seed,
                split_ratio=(
                    (1 - self.holdout_ratio) ** 2,
                    (1 - self.holdout_ratio) * self.holdout_ratio,
                    self.holdout_ratio,
                ),
                pkd_only=False,
                non_pkd_only=True,
                background_classes=self.train_background_classes,
            )

            self.test_dataset = _SmatMrDatasetWithBackgroundInfo(
                root_dir=self.root_dir,
                sequence=sequence,
                target="all",
                stage="test",
                transform=self.test_transform,
                mask_mapping={c: 0 for c in self.test_background_classes},
                cache_rate=self.cache_rate,
                num_workers=self.num_workers,
                dev=self.dev,
                random_seed=self.random_seed,
                split_ratio=(
                    (1 - self.holdout_ratio) ** 2,
                    (1 - self.holdout_ratio) * self.holdout_ratio,
                    self.holdout_ratio,
                ),
                pkd_only=False,
                non_pkd_only=True,
                background_classes=self.test_background_classes,
            )
