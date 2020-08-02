import torch
import torch.nn as nn
from typing import Dict, Any, cast
from mlprogram.nn.utils.rnn import PaddedSequenceWithMask


class Encoder(nn.Module):
    def __init__(self, module: nn.Module, reduction: str = "sum"):
        assert reduction == "sum" or reduction == "mean"
        super().__init__()
        self.module = module
        self.reduction = reduction

    def forward(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        processed_input = cast(torch.Tensor, entry["processed_input"])
        variables = cast(PaddedSequenceWithMask, entry["variables"])
        if len(variables.data) != 0:
            processed_input = \
                processed_input.unsqueeze(0).expand(variables.data.shape)
        else:
            processed_input = torch.zeros_like(variables.data)
        f = torch.cat([processed_input, variables.data], dim=2)
        L, B = f.shape[:2]
        vfeatures = self.module(f.reshape(L * B, *f.shape[2:]))
        vfeatures = vfeatures.reshape(L, B, *vfeatures.shape[1:])

        in_feature = cast(torch.Tensor, entry["input_feature"])
        features = PaddedSequenceWithMask(
            vfeatures * variables.mask.unsqueeze(2),
            variables.mask)
        B = in_feature.shape[0]
        C = in_feature.shape[1:]
        if features.data.numel() == 0:
            features.data = torch.zeros(0, B, *C, device=in_feature.device,
                                        dtype=in_feature.dtype)
        entry["reference_features"] = features

        reduced_feature = features.data.sum(dim=0)
        if self.reduction == "mean":
            in_feature = in_feature.float()
            n_samples = features.mask.sum(dim=0).float().reshape(B, 1)
            reduced_feature = \
                reduced_feature / n_samples
        entry["variable_feature"] = reduced_feature
        entry["input_feature"] = torch.cat([in_feature, reduced_feature],
                                           dim=1)
        return entry
