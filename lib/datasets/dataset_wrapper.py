from __future__ import annotations

import warnings
from abc import ABC
from collections.abc import Callable

from jsonargparse.typing import Path_drw
from monai.data import DataLoader
from monai.data import Dataset as MonaiDataset


class Dataset(ABC):
    """
    A wrapper class to the original dataset class. This class provides a convenient way
    to configure and access dataset-related parameters and data loading options.

    Args:
        in_use (bool): Flag indicating whether the dataset is in use.
        root_dir (Path_drw): The root directory of the dataset.
        num_classes (int): The number of classes in the dataset.
        train_background_classes (list): List of background classes for training.
        test_background_classes (list): List of background classes for testing.
        train_transform (Optional[Callable]): Optional data transformation for training.
        test_transform (Optional[Callable]): Optional data transformation for testing.
        holdout_ratio (float): The ratio of data to hold out for validation.
        cache_rate (float): The rate at which data should be cached.
        num_workers (int): Number of worker processes for data loading.
        train_batch_size (int): Batch size for training data.
        return_datasets (bool): Whether to return the datasets or data loaders.
        dev (bool): Flag indicating whether the code is in development mode.

    Returns:
        If 'return_datasets' is True, returns a tuple of train, validation, and test datasets.
        If 'return_datasets' is False, returns a tuple of train, validation, and test data loaders.
    """

    def __init__(
        self,
        in_use: bool,
        root_dir: Path_drw = "./",
        num_classes: int = 2,
        train_background_classes: list | tuple = (0,),
        test_background_classes: list | tuple = (0,),
        train_transform: Callable | None = None,
        test_transform: Callable | None = None,
        holdout_ratio: float = 0.1,
        cache_rate: float = 0.1,
        num_workers: int = 2,
        train_batch_size: int = 1,
        return_dataloader: bool = True,
        dev: bool = False,
        random_seed: int = 42,
    ):
        self.in_use = in_use

        if self.in_use:
            self.root_dir = root_dir
            self.num_classes = num_classes
            self.train_background_classes = train_background_classes
            self.test_background_classes = test_background_classes
            self.train_transform = train_transform
            self.test_transform = test_transform
            self.holdout_ratio = holdout_ratio
            self.cache_rate = cache_rate
            self.num_workers = num_workers
            self.train_batch_size = train_batch_size
            self.return_dataloader = return_dataloader
            self.dev = dev
            self.random_seed = random_seed

            if not (0 <= holdout_ratio <= 1):
                raise ValueError(f"The value of holdout_ratio is expected to be between 0 and 1, get {holdout_ratio}.")
            if 0 not in train_background_classes:
                raise ValueError("Original background class (0) is not included in train_background_classes")
            if 0 not in test_background_classes:
                raise ValueError("Original background class (0) is not included in test_background_classes")

        self.train_dataset = MonaiDataset({"image": [], "label": []})
        self.val_dataset = MonaiDataset({"image": [], "label": []})
        self.test_dataset = MonaiDataset({"image": [], "label": []})

    def get_data(self):
        if not self.in_use:
            warnings.warn("The dataset is not in use. The method get_data() will return (None, None, None).")
            return None, None, None

        if self.return_dataloader:
            return (
                DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=~self.dev, drop_last=True),
                DataLoader(self.val_dataset, batch_size=1, shuffle=False),
                DataLoader(self.test_dataset, batch_size=1, shuffle=False),
            )
        else:
            return self.train_dataset, self.val_dataset, self.test_dataset
