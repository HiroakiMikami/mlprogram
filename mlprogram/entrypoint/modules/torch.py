from typing import List, Optional, Union

import torch
from torch import nn, optim

from mlprogram import distributed


def device(type_str: str, index: Union[int, str] = 0):
    if index == "rank":
        index = distributed.rank()
    return torch.device(type_str, index)


def Optimizer(optimizer_cls, model: nn.Module, *args, **kwargs) \
        -> optim.Optimizer:
    return optimizer_cls(model.parameters(), *args, **kwargs)


class Reshape(nn.Module):
    def __init__(self, sizes: List[int]):
        super().__init__()
        self.sizes = sizes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(*self.sizes)


class Mean(nn.Module):
    def __init__(self, dim: Optional[int] = None, keepdim: bool = False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.mean(input, dim=self.dim, keepdim=self.keepdim)


def share_memory_(model: nn.Module):
    for k, v in model.state_dict().items():
        v.share_memory_()
    return model


types = {
    "torch.device": device,
    "torch.nn.Sigmoid": lambda: torch.nn.Sigmoid(),
    "torch.Mean": Mean,
    "torch.nn.Sequential": lambda modules: torch.nn.Sequential(modules),
    "torch.optim.Optimizer": Optimizer,
    "torch.optim.Adam": lambda: torch.optim.Adam,
    "torch.optim.SGD": lambda: torch.optim.SGD,
    "torch.Reshape": Reshape,
    "torch.nn.BCELoss": torch.nn.BCELoss,
    "torch.share_memory_": share_memory_
}
