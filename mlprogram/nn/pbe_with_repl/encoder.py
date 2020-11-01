from typing import cast

import torch
import torch.nn as nn

from mlprogram import Environment
from mlprogram.nn.utils.rnn import PaddedSequenceWithMask


class Encoder(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, entry: Environment) -> Environment:
        # (B, N, c)
        processed_input = cast(torch.Tensor, entry.states["test_case_tensor"])
        # (L, B, N, c)
        variables = cast(PaddedSequenceWithMask,
                         entry.states["variables_tensor"])
        # (B, N, C)
        in_feature = cast(torch.Tensor, entry.states["test_case_feature"])

        B, N = in_feature.shape[:2]
        C = in_feature.shape[2:]

        if len(variables.data) != 0:
            # (L, B, N, c)
            processed_input = \
                processed_input.unsqueeze(0).expand(variables.data.shape)
        else:
            # (L, B, N, c)
            processed_input = torch.zeros_like(variables.data)
        # (L, B, N, 2c)
        f = torch.cat([processed_input, variables.data], dim=3)
        L = f.shape[0]
        # (L, B, N, C)
        vfeatures = self.module(f.reshape(L * B * N, *f.shape[3:]))
        vfeatures = vfeatures.reshape(L, B, N, *vfeatures.shape[1:])

        # reduce n_test_cases
        # (L, B, C)
        vfeatures = vfeatures.float().mean(dim=2)
        # (B, C)
        in_feature = in_feature.float().mean(dim=1)

        # Instantiate PaddedSequenceWithMask
        vmask = variables.mask
        for _ in range(len(C)):
            vmask = vmask.reshape(*vmask.shape, 1)
        features = PaddedSequenceWithMask(vfeatures * vmask, variables.mask)
        if features.data.numel() == 0:
            features.data = torch.zeros(0, B, *C, device=in_feature.device,
                                        dtype=in_feature.dtype)

        entry.states["reference_features"] = features

        reduced_feature = features.data.sum(dim=0)  # reduce sequence length
        entry.states["input_feature"] = \
            torch.cat([in_feature, reduced_feature], dim=1)
        return entry
