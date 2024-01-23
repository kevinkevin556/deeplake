from contextlib import nullcontext
from functools import partial
from typing import Optional

import torch
from torch import TensorType, nn


class BaseUpdater:
    """Base class of updaters."""

    def __init__(self):
        pass

    def __call__(self, module):
        self.register_module(module)
        return partial(self.update, module)

    def register_module(self, module):
        self.check_module(module)

    def check_module(self, module: nn.Module) -> None:
        raise NotImplementedError

    def get_alias(self):
        return getattr(self, "alias", self.__class__.__name__)

    def update(
        self,
        module: nn.Module,
        images: TensorType,
        masks: TensorType,
        modalities: Optional[int] = None,
    ) -> float:
        raise NotImplementedError
