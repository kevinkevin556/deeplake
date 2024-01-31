import cv2
from medaset.image_readers import CV2Reader
from monai.data import PydicomReader
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    Resized,
    ScaleIntensityRanged,
    SpatialCropd,
    SpatialPadd,
    ToTensord,
    Transform,
)


class CtTransform(Transform):
    transform = Compose(
        [
            LoadImaged(
                keys=["image", "label"],
                reader=[PydicomReader(swap_ij=False), CV2Reader(flags=cv2.IMREAD_GRAYSCALE)],
            ),
            EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
            ScaleIntensityRanged(keys=["image"], a_min=-200, a_max=200, b_min=0, b_max=1, clip=True),
            CropForegroundd(keys=["image", "label"], source_key="label"),
            SpatialPadd(keys=["image", "label"], spatial_size=(512, 512)),
            SpatialCropd(keys=["image", "label"], roi_center=(256, 256), roi_size=(512, 512)),
            ToTensord(keys=["image", "label"]),
        ]
    )

    def __call__(self, data):
        return self.transform(data)


class MrTransform(Transform):
    transform = Compose(
        [
            LoadImaged(
                keys=["image", "label"],
                reader=[PydicomReader(swap_ij=False, force=True), CV2Reader(flags=cv2.IMREAD_GRAYSCALE)],
                image_only=True,
            ),
            EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
            ScaleIntensityRanged(keys=["image"], a_min=0, a_max=1000, b_min=0, b_max=1, clip=True),
            CropForegroundd(keys=["image", "label"], source_key="label"),
            Resized(keys=["image", "label"], spatial_size=512, size_mode="longest", mode="nearest-exact"),
            SpatialPadd(keys=["image", "label"], spatial_size=(512, 512)),
            ToTensord(keys=["image", "label"]),
        ]
    )

    def __call__(self, data):
        return self.transform(data)


ct_transform = CtTransform()
mr_transform = MrTransform()
