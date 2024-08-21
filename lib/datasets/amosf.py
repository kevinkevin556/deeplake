from __future__ import annotations

from medaset.transforms import ApplyMaskMappingd, BackgroundifyClassesd
from monai.transforms import Compose

from .amos import _AmosDatasetWithBackgroundInfo
from .dataset_wrapper import Dataset


class AmosfCtDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.in_use:
            if self.num_classes != 5:
                raise ValueError(f"There are 5 classes in Simplified AMOS dataset. Got {self.num_classes}")

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
            setattr(self.train_dataset, "num_classes", self.num_classes)
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
            setattr(self.val_dataset, "num_classes", self.num_classes)

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
            setattr(self.test_dataset, "num_classes", self.num_classes)


class AmosfMrDataset(Dataset):
    def __init__(self, *args, sequence: type(None) = None, **kwargs):
        super().__init__(*args, **kwargs)
        if self.in_use:
            if sequence is not None:
                raise ValueError(f"No sequence to determine in Simplified AMOS dataset. Got {sequence}")
            if self.num_classes != 5:
                raise ValueError(f"There are 5 classes in Simplified AMOS dataset. Got {self.num_classes}")

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
            setattr(self.train_dataset, "num_classes", self.num_classes)
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
            setattr(self.val_dataset, "num_classes", self.num_classes)

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
            setattr(self.test_dataset, "num_classes", self.num_classes)
