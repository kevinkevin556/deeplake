import torch


# Dimension hint for pytorch tensors
class tensor:
    def __getitem__(self, *args):
        return torch.Tensor
