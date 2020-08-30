import torch
from torch import nn
from torch import optim
from typing import List, Union
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


types = {
    "torch.device": device,
    "torch.nn.Sigmoid": lambda: torch.nn.Sigmoid(),
    "torch.nn.Sequential": lambda modules: torch.nn.Sequential(modules),
    "torch.optim.Optimizer": Optimizer,
    "torch.optim.Adam": lambda: torch.optim.Adam,
    "torch.Reshape": Reshape,
    "torch.nn.BCELoss": torch.nn.BCELoss,
}
