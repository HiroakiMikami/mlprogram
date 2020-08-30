import torch
import torch.nn as nn
from typing import Optional
from collections import OrderedDict


class MLP(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, hidden_channel: int,
                 n_linear: int, activation: Optional[nn.Module] = None):
        super().__init__()
        assert n_linear > 1
        modules = []
        dim = in_channel
        for i in range(n_linear - 1):
            modules.append((f"linear{i}",
                            nn.Linear(dim, hidden_channel)))
            modules.append((f"act{i}", nn.ReLU()))
            dim = hidden_channel
        modules.append((f"linear{n_linear - 1}", nn.Linear(dim, out_channel)))
        if activation is None:
            modules.append((f"act{n_linear - 1}", nn.ReLU()))
        else:
            modules.append((f"act{n_linear - 1}", activation))
        self.module = nn.Sequential(OrderedDict(modules))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.module(x)
        return out
