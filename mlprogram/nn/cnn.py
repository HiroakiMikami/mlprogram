import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import OrderedDict
from math import prod


def block_2d(in_channel: int, out_channel: int, n_conv: int):
    modules = []
    dim = in_channel
    for i in range(n_conv):
        modules.append((f"conv{i}", nn.Conv2d(dim, out_channel, 3, padding=1)))
        dim = out_channel
        modules.append((f"relu{i}", nn.ReLU()))
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
            dim, hidden_channel, n_conv_per_block)))
        dim = hidden_channel
        for i in range(n_block - 2):
            modules.append((f"block{i + 1}", block_2d(
                dim, hidden_channel, n_conv_per_block)))
            dim = hidden_channel
        modules.append((f"block{n_block - 1}", block_2d(
            dim, out_channel, n_conv_per_block)))
        for name, module in modules:
            self.add_module(name, module)
        self.keys = list([key for key, _ in modules])
        self.pool = pool
        self.flatten = flatten

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        modules = dict(self.named_modules())
        for key in self.keys:
            out = modules[key](out)
            if self.pool != 1 and len(out) != 0:
                out = F.max_pool2d(out, self.pool)
            elif len(out) == 0:
                N, C, H, W = out.shape
                out = out.reshape(N, C, H // self.pool, W // self.pool)
        if self.flatten:
            return out.reshape(out.shape[0], prod(out.shape[1:]))
        else:
            return out
