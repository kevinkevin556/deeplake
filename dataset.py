import os
from glob import glob

import monai
import numpy as np
import torch
from monai.transforms import (
    Activationsd,
    AddChanneld,
    AsDiscreted,
    Compose,
    CropForegroundd,
    EnsureTyped,
    Invertd,
    KeepLargestConnectedComponentd,
    LoadImaged,
    Orientationd,
    RandAffined,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    ResizeWithPadOrCropd,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    ToTensord,
)

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
        crop_samples=2,
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

        if transform:
            pass
        elif stage == "train":
            transform = Compose(
                [
                    LoadImaged(keys=["image", "label"]),
                    AddChanneld(keys=["image", "label"]),
                    Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
                    Orientationd(keys=["image", "label"], axcodes="RAS"),
                    ScaleIntensityRanged(
                        keys=["image"],
                        a_min=-125,
                        a_max=275,
                        b_min=0.0,
                        b_max=1.0,
                        clip=True,
                    ),
                    CropForegroundd(keys=["image", "label"], source_key="image"),
                    RandCropByPosNegLabeld(
                        keys=["image", "label"],
                        label_key="label",
                        spatial_size=(96, 96, 96),
                        pos=1,
                        neg=1,
                        num_samples=crop_samples,
                        image_key="image",
                        image_threshold=0,
                        allow_smaller=True,  ### MOD ###
                    ),
                    SpatialPadd(keys=["image", "label"], spatial_size=(96, 96, 96)),  ### MOD ###
                    RandShiftIntensityd(
                        keys=["image"],
                        offsets=0.10,
                        prob=0.50,
                    ),
                    RandAffined(
                        keys=["image", "label"],
                        mode=("bilinear", "nearest"),
                        prob=1.0,
                        spatial_size=(96, 96, 96),
                        rotate_range=(0, 0, np.pi / 30),
                        scale_range=(0.1, 0.1, 0.1),
                    ),
                    ToTensord(keys=["image", "label"]),
                ]
            )
        elif (stage == "validation") or (stage == "test"):
            transform = Compose(
                [
                    LoadImaged(keys=["image", "label"]),
                    AddChanneld(keys=["image", "label"]),
                    Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
                    Orientationd(keys=["image", "label"], axcodes="RAS"),
                    ScaleIntensityRanged(
                        keys=["image"],
                        a_min=-125,
                        a_max=275,
                        b_min=0.0,
                        b_max=1.0,
                        clip=True,
                    ),
                    CropForegroundd(keys=["image", "label"], source_key="image"),
                    ToTensord(keys=["image", "label"]),
                ]
            )
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

    # def __getitem__(self, idx):
    #     sample = super().__getitem__(idx)
    #     image = sample["image"].as_tensor()
    #     label = sample["label"].as_tensor().long()
    #     if self.spatial_dim == 2:
    #         return image[:, :, :, 0], label[0, :, :, 0]
    #     else:
    #         return image, label


class CHAOSDataset:
    def __init__(self, root_dir, modality, stage="train", spatial_dim=2, transform=True, dev=False):
        self.modality = modality
        self.stage = stage
        self.spatial_dim = spatial_dim

        if self.modality == "ct":
            self.image_path = [
                os.path.join(root_dir, p)
                for p in glob(os.path.join(root_dir, "CT/*/DICOM_anon/*.dcm"))
                if p.name != ".DS_Store"
            ]
        self.target_path = [
            os.path.join(root_dir, p)
            for p in os.scandir(os.path.join(root_dir, f"labels{abbr[self.stage]}"))
            if p.name != ".DS_Store"
        ]

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
