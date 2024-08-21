import numpy as np
import torch


# Dimension hint for pytorch tensors
class tensor:
    def __class_getitem__(cls, *args):
        return torch.Tensor

    def __getitem__(self, *args):
        return torch.Tensor


# Dimension hint for numpy ndarrays
class array:
    def __class_getitem__(cls, *args):
        return np.ndarray

    def __getitem__(self, *args):
        return np.ndarray
