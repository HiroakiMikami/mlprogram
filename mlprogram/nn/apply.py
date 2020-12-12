from typing import Any, Dict, List, Tuple, Union

from torch import nn

from mlprogram import Environment


class Apply(nn.Module):
    def __init__(self, in_keys: List[Union[str, Tuple[str, str]]],
                 out_key: Union[str, List[str]],
                 module: nn.Module,
                 constants: Dict[str, Any] = {}):
        super().__init__()
        self.in_keys = in_keys
        self.out_key = out_key
        self.module = module
        self.constants = constants

    def forward(self, entry: Environment) -> Environment:
        kwargs = {key: value for key, value in self.constants.items()}
        for i, in_key in enumerate(self.in_keys):
            if isinstance(in_key, str):
                original_key = in_key
                renamed_key = in_key
            else:
                original_key, renamed_key = in_key
            kwargs[renamed_key] = entry[original_key]

        output = self.module(**kwargs)
        if isinstance(self.out_key, str):
            entry[self.out_key] = output
        else:
            assert len(self.out_key) == len(output)
            for key, out in zip(self.out_key, output):
                entry[key] = out
        return entry
