from typing import Any

from torch import nn

from mlprogram.builtins.datatypes import Environment


class Pack(nn.Module):
    def __init__(self, key: str, is_supervision: bool = False):
        super().__init__()
        self._key = key
        self._is_supervision = is_supervision

    def forward(self, value: Any) -> Environment:
        out = Environment(
            {self._key: value}
        )
        if self._is_supervision:
            out.mark_as_supervision(self._key)
        return out
