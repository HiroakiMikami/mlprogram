import torch
import torch.nn as nn
from typing import Dict, Any, cast, List
from mlprogram.nn.utils.rnn import pad_sequence


class Encoder(nn.Module):
    def __init__(self, module: nn.Module,
                 reduction: str = "sum"):
        assert reduction == "sum" or reduction == "mean"
        super().__init__()
        self.module = module
        self.reduction = reduction

    def forward(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        input = cast(torch.Tensor, entry["input"])
        variables = cast(List[torch.Tensor], entry["variables"])
        in_feature = self.module(input)
        C = in_feature.shape[1:]
        sizes = [len(v) for v in variables]
        if sum(sizes) == 0:
            features = [torch.zeros(0, *C, device=in_feature.device,
                                    dtype=in_feature.dtype) for _ in variables]
        else:
            packed_features = self.module(torch.cat(variables, dim=0))
            features = torch.split(packed_features, tuple(sizes))
        entry["reference_features"] = pad_sequence(features)

        if self.reduction == "sum":
            reduced_feature = torch.stack([f.sum(dim=0) for f in features])
        else:
            reduced_feature = torch.stack([f.mean(dim=0) for f in features])
        entry["input_feature"] = torch.cat([in_feature, reduced_feature],
                                           dim=1)
        return entry
