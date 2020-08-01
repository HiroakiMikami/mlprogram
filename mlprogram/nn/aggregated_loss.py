import torch
from torch import nn
from typing import Dict, Any
from pytorch_pfn_extras.reporting import report


class AggregatedLoss(nn.Module):
    def __init__(self, losses: Dict[str, nn.Module]):
        super().__init__()
        self.losses = losses

    def forward(self, entry: Dict[str, Any]) -> torch.Tensor:
        losses = {key: loss(entry).reshape(1)
                  for key, loss in self.losses.items()}
        report({key: loss.item() for key, loss in losses.items()})
        return torch.sum(torch.cat(list(losses.values())))
