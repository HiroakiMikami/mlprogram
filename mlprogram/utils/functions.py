import torch
import os
from collections import OrderedDict
from typing import Generic, TypeVar, Optional, Any, List, Callable, Dict
from mlprogram.utils import logging

logger = logging.Logger(__name__)

V = TypeVar("V")
V0 = TypeVar("V0")
V1 = TypeVar("V1")


class Compose:
    def __init__(self, funcs: OrderedDict):
        self.funcs = funcs

    @logger.function_block("Compose.__call__")
    def __call__(self, value: Optional[Any]) -> Optional[Any]:
        if value is None:
            return None
        for key, f in self.funcs.items():
            with logger.block(key):
                value = f(value)
                if value is None:
                    return None
        return value


class Sequence:
    def __init__(self, funcs: OrderedDict):
        self.funcs = funcs

    @logger.function_block("Sequence.__call__")
    def __call__(self, values: Any) -> Optional[Any]:
        value_opt: Optional[Any] = values
        for key, func in self.funcs.items():
            with logger.block(key):
                value_opt = func(value_opt)
                if value_opt is None:
                    return None
        return value_opt


class Map(Generic[V0, V1]):
    def __init__(self, func: Callable[[V0], V1]):
        self.func = func

    @logger.function_block("Map.__call__")
    def __call__(self, values: List[V0]) -> List[V1]:
        return [self.func(v0)
                for v0 in logger.iterable_block("values", values)]


class Flatten(Generic[V]):
    def __call__(self, values: List[List[V]]) -> List[V]:
        retval = []
        for v in values:
            retval.extend(v)
        return retval


class Threshold(object):
    def __init__(self, threshold: float, dtype: str = "bool"):
        self.threshold = threshold
        assert dtype in set(["bool", "int", "float"])
        if dtype == "bool":
            self.dtype: Callable[[bool], Any] = bool
        elif dtype == "int":
            self.dtype = int
        elif dtype == "float":
            self.dtype = float

    def __call__(self, value: float) -> bool:
        out = value >= self.threshold
        return self.dtype(out)


class Pick(object):
    def __init__(self, key: str):
        self.key = key

    def __call__(self, entry: Dict[str, Any]) -> Optional[Any]:
        return entry[self.key] if self.key in entry else None


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
