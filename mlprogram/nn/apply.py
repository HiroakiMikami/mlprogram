import torch
from torch import nn
from typing import Dict, Any


class Apply(nn.Module):
    def __init__(self, in_key: str, out_key: str,
                 module: nn.Module,
                 is_sequence: bool = False):
        super().__init__()
        self.in_key = in_key
        self.out_key = out_key
        self.module = module
        self.is_sequence = is_sequence

    def forward(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        if self.is_sequence:
            input = entry[self.in_key]
            sizes = tuple([len(x) for x in input])
            if sum(sizes) != 0:
                packed_input = torch.cat(input, dim=0)
                packed_output = self.module(packed_input)
                output = torch.split(packed_output, sizes)
                entry[self.out_key] = output
            else:
                entry[self.out_key] = [[] for _ in input]
        else:
            input = entry[self.in_key]
            output = self.module(input)
            entry[self.out_key] = output
        return entry
