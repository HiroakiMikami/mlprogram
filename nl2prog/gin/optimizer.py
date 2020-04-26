from torch import optim
from torch import nn
from typing import Callable, Any


def create_optimizer(optimizer_cls: Callable[[Any], optim.Optimizer],
                     model: nn.Module) -> optim.Optimizer:
    return optimizer_cls(model.parameters())
