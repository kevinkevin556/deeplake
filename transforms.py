import numpy as np
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

amos_train_transforms = Compose(
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
            num_samples=2,
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


amos_val_transforms = Compose(
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
