import torch
import logging
import os
from collections import OrderedDict
from typing import Generic, TypeVar, Optional, Any, List, Callable

logger = logging.getLogger(__name__)

V = TypeVar("V")
V0 = TypeVar("V0")
V1 = TypeVar("V1")


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


class Map(Generic[V0, V1]):
    def __init__(self, func: Callable[[V0], V1]):
        self.func = func

    def __call__(self, values: List[V0]) -> List[V1]:
        return [self.func(v0) for v0 in values]


class Flatten(Generic[V]):
    def __call__(self, values: List[List[V]]) -> List[V]:
        retval = []
        for v in values:
            retval.extend(v)
        return retval


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
