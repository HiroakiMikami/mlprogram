import torch
from torch import nn
from typing import Dict, Any
from enum import Enum
from mlprogram.nn.utils.rnn import PaddedSequenceWithMask


class ApplyOptions(Enum):
    Nothing = 1,
    Sequence = 2,
    PaddedSequence = 3,


class Apply(nn.Module):
    def __init__(self, in_key: str, out_key: str,
                 module: nn.Module,
                 options: ApplyOptions = ApplyOptions.Nothing):
        super().__init__()
        self.in_key = in_key
        self.out_key = out_key
        self.module = module
        self.options = options

    def forward(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        if self.options == ApplyOptions.Nothing:
            input = entry[self.in_key]
            output = self.module(input)
            entry[self.out_key] = output
        elif self.options == ApplyOptions.Sequence:
            input = entry[self.in_key]
            sizes = tuple([len(x) for x in input])
            if sum(sizes) != 0:
                packed_input = torch.cat(input, dim=0)
                packed_output = self.module(packed_input)
                output = torch.split(packed_output, sizes)
                entry[self.out_key] = output
            else:
                entry[self.out_key] = [[] for _ in input]
        elif self.options == ApplyOptions.PaddedSequence:
            input = entry[self.in_key]
            L, B = input.data.shape[:2]
            output = self.module(input.data.reshape(L * B, *input.data.shape[2:]))
            output = output.reshape(L, B, *output.shape[1:])
            entry[self.out_key] = PaddedSequenceWithMask(output, input.mask)
        return entry
