from torch import nn
from typing import Callable


class Function(nn.Module):
    def __init__(self, f: Callable):
        super().__init__()
        self.f = f

    def forward(self, *args, **kwargs):
        return self.f(*args, **kwargs)
