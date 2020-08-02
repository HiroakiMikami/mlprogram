import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import OrderedDict
import numpy as np


class MaxPool2d(nn.Module):
    def __init__(self, pool: int):
        super().__init__()
        self.pool = pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x) != 0:
            return F.max_pool2d(x, self.pool)
        elif len(x) == 0:
            N, C, H, W = x.shape
            return x.reshape(N, C, H // self.pool, W // self.pool)


def block_2d(in_channel: int, out_channel: int, hidden_channel: int,
             n_conv: int, pool: int):
    modules = []
    dim = in_channel
    for i in range(n_conv - 1):
        modules.append((f"conv{i}",
                        nn.Conv2d(dim, hidden_channel, 3, padding=1)))
        dim = hidden_channel
        modules.append((f"relu{i}", nn.ReLU()))
    modules.append((f"conv{n_conv - 1}",
                    nn.Conv2d(dim, out_channel, 3, padding=1)))
    modules.append((f"relu{n_conv - 1}", nn.ReLU()))
    if pool != 1:
        modules.append(("pool", MaxPool2d(pool)))
    return nn.Sequential(OrderedDict(modules))


class CNN2d(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, hidden_channel: int,
                 n_conv_per_block: int, n_block: int, pool: int,
                 flatten: bool = True):
        super().__init__()
        assert n_block >= 2
        modules = []
        modules.append(("block0", block_2d(
            in_channel, hidden_channel, hidden_channel, n_conv_per_block,
            pool)))
        dim = hidden_channel
        for i in range(n_block - 2):
            modules.append((f"block{i + 1}", block_2d(
                dim, hidden_channel, hidden_channel, n_conv_per_block, pool)))
            dim = hidden_channel
        modules.append((f"block{n_block - 1}", block_2d(
            dim, out_channel, hidden_channel, n_conv_per_block, 1)))
        self.module = torch.nn.Sequential(OrderedDict(modules))
        self.flatten = flatten

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.module(x)
        if self.flatten:
            return out.reshape(out.shape[0], np.prod(out.shape[1:]))
        else:
            return out
