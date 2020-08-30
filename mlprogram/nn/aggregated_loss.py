import torch
from torch import nn
from pytorch_pfn_extras.reporting import report


class AggregatedLoss(nn.Module):
    def forward(self, **kwargs: torch.Tensor) -> torch.Tensor:
        losses = {key: loss.sum().reshape(1) for key, loss in kwargs.items()}
        report({key: loss.item() for key, loss in losses.items()})
        return torch.sum(torch.cat(list(losses.values())))
