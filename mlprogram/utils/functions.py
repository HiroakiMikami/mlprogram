import torch
import logging
import os
from collections import OrderedDict
from typing import TypeVar, Optional, Any

logger = logging.getLogger(__name__)

V = TypeVar("V")


class Compose:
    def __init__(self, funcs: OrderedDict):
        self.funcs = funcs

    def __call__(self, value: Optional[Any]) -> Optional[Any]:
        if value is None:
            return None
        for f in self.funcs.values():
            value = f(value)
            if value is None:
                return None
        return value


class Sequence:
    def __init__(self, funcs: OrderedDict):
        self.funcs = funcs

    def __call__(self, values: Any) -> Optional[Any]:
        value_opt: Optional[Any] = values
        for func in self.funcs.values():
            value_opt = func(value_opt)
            if value_opt is None:
                return None
        return value_opt


def save(obj: V, file: str) -> V:
    if os.path.exists(file):
        logger.info(f"Reuse data from {file}")
        return torch.load(file)

    os.makedirs(os.path.dirname(file), exist_ok=True)
    torch.save(obj, file)
    return obj


def load(file: str) -> Any:
    logger.info(f"Load data from {file}")
    return torch.load(file)
