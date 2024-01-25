from __future__ import annotations

from medaset.transforms import ApplyMaskMappingd, BackgroundifyClassesd
from monai.transforms import Compose

from .amos import _AmosDatasetWithBackgroundInfo
from .dataset_wrapper import Dataset


class SimpleAmosCtDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_transform = Compose(
            [
                self.train_transform,
                BackgroundifyClassesd(keys=["label"], channel_dim=0, classes=[8, 9, 11, 12, 13, 14, 15]),
                ApplyMaskMappingd(keys=["label"], mask_mapping={10: 8}),
            ]
        )
        self.test_transforms = Compose(
            [
                self.test_transforms,
                BackgroundifyClassesd(keys=["label"], channel_dim=0, classes=[8, 9, 11, 12, 13, 14, 15]),
                ApplyMaskMappingd(keys=["label"], mask_mapping={10: 8}),
            ]
        )
        if self.num_classes != 9:
            raise ValueError(f"There are 9 classes in Simplified AMOS dataset. Got {self.num_classes}")

        self.train_dataset = _AmosDatasetWithBackgroundInfo(
            root_dir=self.root_dir,
            modality="ct",
            stage="train",
            transform=self.train_transform,
            mask_mapping={c: 0 for c in self.train_background_classes},
            cache_rate=self.cache_rate,
            num_workers=self.num_workers,
            background_classes=self.train_background_classes,
        )
        setattr(self.train_dataset, "batch_size", self.train_batch_size)
        self.train_dataset = self.train_dataset[: -int(len(self.train_dataset) * self.holdout_ratio)]

        self.val_dataset = _AmosDatasetWithBackgroundInfo(
            root_dir=self.root_dir,
            modality="ct",
            stage="val",
            transform=self.test_transforms,
            mask_mapping={c: 0 for c in self.train_background_classes},
            cache_rate=self.cache_rate,
            num_workers=self.num_workers,
            background_classes=self.train_background_classes,
        )
        self.val_dataset = self.val_dataset[-int(len(self.val_dataset) * self.holdout_ratio) :]

        self.test_dataset = _AmosDatasetWithBackgroundInfo(
            root_dir=self.root_dir,
            modality="ct",
            stage="test",
            transform=self.test_transforms,
            mask_mapping={c: 0 for c in self.test_background_classes},
            cache_rate=self.cache_rate,
            num_workers=self.num_workers,
            background_classes=self.test_background_classes,
        )


class SimpleAmosMrDataset(Dataset):
    def __init__(self, *args, sequence: type(None) = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_transform = Compose(
            [
                self.train_transform,
                BackgroundifyClassesd(keys=["label"], channel_dim=0, classes=[8, 9, 11, 12, 13, 14, 15]),
                ApplyMaskMappingd(keys=["label"], mask_mapping={10: 8}),
            ]
        )
        self.test_transforms = Compose(
            [
                self.test_transforms,
                BackgroundifyClassesd(keys=["label"], channel_dim=0, classes=[8, 9, 11, 12, 13, 14, 15]),
                ApplyMaskMappingd(keys=["label"], mask_mapping={10: 8}),
            ]
        )
        if sequence is not None:
            raise ValueError(f"No sequence to determine in Simplified AMOS dataset. Got {sequence}")
        if self.num_classes != 9:
            raise ValueError(f"There are 9 classes in Simplified AMOS dataset. Got {self.num_classes}")

        self.train_dataset = _AmosDatasetWithBackgroundInfo(
            root_dir=self.root_dir,
            modality="mr",
            stage="train",
            transform=self.train_transform,
            mask_mapping={c: 0 for c in self.train_background_classes},
            cache_rate=self.cache_rate,
            num_workers=self.num_workers,
            background_classes=self.train_background_classes,
        )
        setattr(self.train_dataset, "batch_size", self.train_batch_size)
        self.train_dataset = self.train_dataset[: -int(len(self.train_dataset) * self.holdout_ratio)]

        self.val_dataset = _AmosDatasetWithBackgroundInfo(
            root_dir=self.root_dir,
            modality="mr",
            stage="val",
            transform=self.test_transforms,
            mask_mapping={c: 0 for c in self.train_background_classes},
            cache_rate=self.cache_rate,
            num_workers=self.num_workers,
            background_classes=self.train_background_classes,
        )
        self.val_dataset = self.val_dataset[-int(len(self.val_dataset) * self.holdout_ratio) :]

        self.test_dataset = _AmosDatasetWithBackgroundInfo(
            root_dir=self.root_dir,
            modality="mr",
            stage="test",
            transform=self.test_transforms,
            mask_mapping={c: 0 for c in self.test_background_classes},
            cache_rate=self.cache_rate,
            num_workers=self.num_workers,
            background_classes=self.test_background_classes,
        )
