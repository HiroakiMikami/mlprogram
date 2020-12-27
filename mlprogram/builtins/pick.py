from typing import Any, Optional

from torch import nn

from mlprogram.builtins.datatypes import Environment


class Pick(nn.Module):
    def __init__(self, key: str):
        super().__init__()
        self.key = key

    def forward(self, entry: Environment) -> Optional[Any]:
        return entry[self.key] if self.key in entry.to_dict() else None
