from jsonargparse.typing import path_type
from medaset import amos
from medaset.transforms import ApplyMaskMappingd, BackgroundifyClassesd
from monai.transforms import Compose

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
        assert self.num_classes == 9

        self.train_dataset = amos.AmosDataset(
            root_dir=self.root_dir,
            modality="ct",
            stage="train",
            transform=self.train_transform,
            mask_mapping={c: 0 for c in self.train_background_classes},
            cache_rate=self.cache_rate,
            num_workers=self.num_workers,
        )
        setattr(self.train_dataset, "batch_size", self.train_batch_size)

        self.val_dataset = amos.AmosDataset(
            root_dir=self.root_dir,
            modality="ct",
            stage="val",
            transform=self.test_transforms,
            mask_mapping={c: 0 for c in self.train_background_classes},
            cache_rate=self.cache_rate,
            num_workers=self.num_workers,
        )

        self.test_dataset = amos.AmosDataset(
            root_dir=self.root_dir,
            modality="ct",
            stage="test",
            transform=self.test_transforms,
            mask_mapping={c: 0 for c in self.test_background_classes},
            cache_rate=self.cache_rate,
            num_workers=self.num_workers,
        )


class SimpleAmosMrDataset(Dataset):
    def __init__(self, sequence: type(None) = None, *args, **kwargs):
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
        assert self.num_classes == 9

        self.train_dataset = amos.AmosDataset(
            root_dir=self.root_dir,
            modality="mr",
            stage="train",
            transform=self.train_transform,
            mask_mapping={c: 0 for c in self.train_background_classes},
            cache_rate=self.cache_rate,
            num_workers=self.num_workers,
        )
        setattr(self.train_dataset, "batch_size", self.train_batch_size)

        self.val_dataset = amos.AmosDataset(
            root_dir=self.root_dir,
            modality="mr",
            stage="val",
            transform=self.test_transforms,
            mask_mapping={c: 0 for c in self.train_background_classes},
            cache_rate=self.cache_rate,
            num_workers=self.num_workers,
        )

        self.test_dataset = amos.AmosDataset(
            root_dir=self.root_dir,
            modality="mr",
            stage="test",
            transform=self.test_transforms,
            mask_mapping={c: 0 for c in self.test_background_classes},
            cache_rate=self.cache_rate,
            num_workers=self.num_workers,
        )
