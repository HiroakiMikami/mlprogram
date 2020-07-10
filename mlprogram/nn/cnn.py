import torch
import torch.nn as nn
from collections import OrderedDict


def block_2d(in_channel: int, out_channel: int, n_conv: int, pool: int):
    modules = []
    dim = in_channel
    for i in range(n_conv):
        modules.append((f"conv{i}", nn.Conv2d(dim, out_channel, 3, padding=1)))
        dim = out_channel
        modules.append((f"relu{i}", nn.ReLU()))
    if pool != 1:
        modules.append(("pool", nn.MaxPool2d(2)))
    return nn.Sequential(OrderedDict(modules))


class CNN2d(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, hidden_channel: int,
                 n_conv_per_block: int, n_block: int, pool: int,
                 flatten: bool = True):
        super().__init__()
        assert n_block >= 2
        modules = []
        dim = in_channel
        modules.append(("block0", block_2d(
            dim, hidden_channel, n_conv_per_block, pool)))
        dim = hidden_channel
        for i in range(n_block - 2):
            modules.append((f"block{i + 1}", block_2d(
                dim, hidden_channel, n_conv_per_block, pool)))
            dim = hidden_channel
        modules.append((f"block{n_block - 1}", block_2d(
            dim, out_channel, n_conv_per_block, pool)))
        self.module = nn.Sequential(OrderedDict(modules))
        self.flatten = flatten

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.module(x)
        if self.flatten:
            return out.reshape(out.shape[0], -1)
        else:
            return out
