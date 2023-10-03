import os
from glob import glob

import monai
import numpy as np
import torch

from transforms import amos_train_transforms, amos_val_transforms

abbr = {"train": "Tr", "validation": "Va", "test": "Ts"}


def get_file_number(filename):
    return int(str(filename).split("_")[-1].split(".")[0])


class AMOSDataset(monai.data.CacheDataset):
    def __init__(
        self,
        root_dir,
        modality="ct",
        stage="train",
        spatial_dim=3,
        transform=None,
        dev=False,
        cache_rate=0.1,
        num_workers=2,
    ):
        self.modality = modality
        self.stage = stage
        self.spatial_dim = spatial_dim

        self.image_path = [
            os.path.join(root_dir, p)
            for p in os.scandir(os.path.join(root_dir, f"images{abbr[self.stage]}"))
            if p.name != ".DS_Store"
        ]
        self.target_path = [
            os.path.join(root_dir, p)
            for p in os.scandir(os.path.join(root_dir, f"labels{abbr[self.stage]}"))
            if p.name != ".DS_Store"
        ]
        assert len(self.image_path) == len(self.target_path)

        # Collect data with specified modality
        if self.modality == "ct":
            self.image_path = [p for p in self.image_path if get_file_number(p) <= 500]
            self.target_path = [p for p in self.target_path if get_file_number(p) <= 500]
        elif self.modality == "mr":
            self.image_path = [p for p in self.image_path if get_file_number(p) > 500]
            self.target_path = [p for p in self.target_path if get_file_number(p) > 500]
        elif self.modality == "ct+mr":
            pass
        else:
            raise ValueError("Invalid modality is specified. Options are {ct, mr, ct+mr}.")

        # Developing mode (20% for training data, 5% for other stage)
        if dev:
            if stage == "train":
                n_train_dev = max(int(len(self.image_path) * 0.2), 40)
                self.image_path = self.image_path[:n_train_dev]
                self.target_path = self.target_path[:n_train_dev]
            else:
                n_val_dev = max(int(len(self.image_path) * 0.02), 5)
                self.image_path = self.image_path[:n_val_dev]
                self.target_path = self.target_path[:n_val_dev]

        # Transformation
        if transform is not None:
            pass
        elif stage == "train":
            transform = amos_train_transforms
        elif (stage == "validation") or (stage == "test"):
            transform = amos_val_transforms
        else:
            raise ValueError("Either stage or transform should be specified.")

        super().__init__(
            data=[{"image": im, "label": la} for im, la in zip(self.image_path, self.target_path)],
            transform=transform,
            cache_rate=cache_rate,
            num_workers=num_workers,
        )

    def __len__(self):
        return len(self.target_path)


class CHAOSDataset:
    def __init__(self, root_dir, modality, stage="train", spatial_dim=2, transform=True, dev=False):
        self.modality = modality
        self.stage = stage
        self.spatial_dim = spatial_dim

        if self.modality == "ct":
            self.image_path = sorted(
                [
                    os.path.join(root_dir, p)
                    for p in glob(os.path.join(root_dir, "CT/*/DICOM_anon/*.dcm"))
                    if p.name != ".DS_Store"
                ]
            )
        self.target_path = sorted(
            [
                os.path.join(root_dir, p)
                for p in os.scandir(os.path.join(root_dir, f"labels{abbr[self.stage]}"))
                if p.name != ".DS_Store"
            ]
        )

        if dev:
            # Training using only 20 images
            self.image_path = self.image_path[:20]
            self.target_path = self.target_path[:20]

        assert len(self.image_path) == len(self.target_path)

        if self.modality == "ct":
            self.image_path = [p for p in self.image_path if get_file_number(p) <= 500]
            self.target_path = [p for p in self.target_path if get_file_number(p) <= 500]
        elif self.modality == "mr":
            self.image_path = [p for p in self.image_path if get_file_number(p) > 500]
            self.target_path = [p for p in self.target_path if get_file_number(p) > 500]
        else:
            # modality == "all"
            pass
