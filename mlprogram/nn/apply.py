from typing import Any, Dict, List, Tuple, Union

import torch
from torch import nn

from mlprogram import Environment
from mlprogram.nn.utils.rnn import PaddedSequenceWithMask


class Apply(nn.Module):
    def __init__(self, in_keys: List[Union[str, Tuple[str, str]]],
                 out_key: str,
                 module: nn.Module,
                 value_type: str = "tensor",
                 constants: Dict[str, Any] = {}):
        super().__init__()
        self.in_keys = in_keys
        self.out_key = out_key
        self.module = module
        assert value_type in set(["tensor", "list", "padded_tensor"])
        self.value_type = value_type
        self.constants = constants

    def forward(self, entry: Environment) -> Environment:
        kwargs = {key: value for key, value in self.constants.items()}
        for i, in_key in enumerate(self.in_keys):
            if isinstance(in_key, str):
                original_key = in_key
                _, renamed_key = Environment.parse_key(in_key)
            else:
                original_key, renamed_key = in_key
            kwargs[renamed_key] = entry[original_key]
            if i == 0:
                main_key = renamed_key

        if self.value_type == "tensor":
            output = self.module(**kwargs)
            entry[self.out_key] = output
        elif self.value_type == "list":
            input = kwargs[main_key]
            sizes = tuple([len(x) for x in input])
            if sum(sizes) != 0:
                packed_input = torch.cat(input, dim=0)
                kwargs[main_key] = packed_input
                packed_output = self.module(**kwargs)
                output = torch.split(packed_output, sizes)
                entry[self.out_key] = output
            else:
                entry[self.out_key] = [[] for _ in input]
        elif self.value_type == "padded_tensor":
            input = kwargs[main_key]
            L, B = input.data.shape[:2]
            kwargs[main_key] = input.data.reshape(L * B, *input.data.shape[2:])
            output = self.module(**kwargs)
            output = output.reshape(L, B, *output.shape[1:])
            entry[self.out_key] = PaddedSequenceWithMask(output, input.mask)
        return entry
