import torch
from torch import nn
from typing import Dict, Any


class AggregatedLoss(nn.Module):
    def __init__(self, losses: Dict[str, nn.Module]):
        super().__init__()
        self.losses = losses

    def forward(self, entry: Dict[str, Any]) -> torch.Tensor:
        return torch.sum(torch.cat([
            loss(entry).reshape(1) for _, loss in self.losses.items()
        ]))
