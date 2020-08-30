from torch import nn
from typing import Dict, Any, Optional


class Pick(nn.Module):
    def __init__(self, key: str):
        super().__init__()
        self.key = key

    def forward(self, entry: Dict[str, Any]) -> Optional[Any]:
        return entry[self.key] if self.key in entry else None
