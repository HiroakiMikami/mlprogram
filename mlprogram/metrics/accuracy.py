from typing import Generic, TypeVar

from torch import nn

Value = TypeVar("Value")


class Accuracy(nn.Module, Generic[Value]):
    def forward(self, expected: Value, actual: Value) -> float:
        return 1.0 if actual == expected else 0.0
