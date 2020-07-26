import torch
import torch.nn as nn
from typing import Dict, Any, cast
from mlprogram.nn.utils.rnn import PaddedSequenceWithMask


class Encoder(nn.Module):
    def __init__(self, reduction: str = "sum"):
        assert reduction == "sum" or reduction == "mean"
        super().__init__()
        self.reduction = reduction

    def forward(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        in_feature = cast(torch.Tensor, entry["input_feature"])
        features = cast(PaddedSequenceWithMask, entry["reference_features"])
        B = in_feature.shape[0]
        C = in_feature.shape[1:]
        if features.data.numel() == 0:
            features.data = torch.zeros(0, B, *C, device=in_feature.device,
                                        dtype=in_feature.dtype)
        entry["reference_features"] = features

        reduced_feature = features.data.sum(dim=0)
        if self.reduction == "mean":
            in_feature = in_feature.float()
            reduced_feature = \
                reduced_feature / features.mask.sum(dim=0).float()
        entry["input_feature"] = torch.cat([in_feature, reduced_feature],
                                           dim=1)
        return entry
