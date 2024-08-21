from __future__ import annotations

import warnings
from functools import partial

from torch import TensorType, nn


class BaseUpdater:
    """Base class of updaters."""

    def __init__(self):
        pass

    def __call__(self, module):
        return self.register_module(module)

    def register_module(self, module):
        self.check_module(module)
        return partial(self.update, module)

    def check_module(self, module: nn.Module) -> None:
        # if not getattr(module, "criterion", False):
        #     warnings.warn("There is no `criterion` in module.", UserWarning)
        # if not getattr(module, "optimizer", False):
        #     warnings.warn("There is no `optimizer` in module.", UserWarning)
        pass

    def get_alias(self):
        return getattr(self, "alias", self.__class__.__name__)

    def update(self, module: nn.Module, images: TensorType, targets: TensorType, **kwargs) -> float:
        module.optimizer.zero_grad()
        preds = module(images)
        loss = module.criterion(preds, targets)
        loss.backward()
        module.optimizer.step()
        return loss.item()
