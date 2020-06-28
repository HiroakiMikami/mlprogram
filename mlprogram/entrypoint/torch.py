import torch
from torch import nn
from torch import optim


def device(type_str: str, index: int = 0):
    return torch.device(type_str, index)


def create_optimizer(optimizer_cls,
                     model: nn.Module) -> optim.Optimizer:
    return optimizer_cls(model.parameters())


types = {
    "torch.device": device,
    "torch.nn.Sequential": lambda modules: torch.nn.Sequential(modules),
    "torch.optim.create_optimizer": create_optimizer,
    "torch.optim.Adam": lambda: torch.optim.Adam
}
