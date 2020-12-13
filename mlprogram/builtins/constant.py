from typing import Any

from torch import nn


class Constant(nn.Module):
    def __init__(self, value: Any):
        super().__init__()
        self._value = value

    def forward(self) -> Any:
        return self._value
