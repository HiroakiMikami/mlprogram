import torch
import torch.nn as nn
from collections import OrderedDict


def block(in_channel: int, out_channel: int, n_linear: int):
    modules = []
    dim = in_channel
    for i in range(n_linear):
        modules.append((f"linear{i}", nn.Linear(dim, out_channel)))
        dim = out_channel
        modules.append((f"relu{i}", nn.ReLU()))
    return nn.Sequential(OrderedDict(modules))


class MLP(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, hidden_channel: int,
                 n_linear_per_block: int, n_block: int):
        super().__init__()
        assert n_block >= 2
        modules = []
        dim = in_channel
        modules.append(("block0", block(
            dim, hidden_channel, n_linear_per_block)))
        dim = hidden_channel
        for i in range(n_block - 2):
            modules.append((f"block{i + 1}", block(
                dim, hidden_channel, n_linear_per_block)))
            dim = hidden_channel
        modules.append((f"block{n_block - 1}", block(
            dim, out_channel, n_linear_per_block)))
        self.module = nn.Sequential(OrderedDict(modules))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)
