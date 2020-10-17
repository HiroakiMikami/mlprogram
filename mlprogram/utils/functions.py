from torch import nn
from torch import multiprocessing
from collections import OrderedDict
from typing import Generic, TypeVar, Optional, Any, List, Callable
from mlprogram import Environment
from mlprogram import logging

logger = logging.Logger(__name__)

V = TypeVar("V")
V0 = TypeVar("V0")
V1 = TypeVar("V1")


class Identity(Generic[V]):
    def __call__(self, value: V) -> V:
        return value


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
    def __init__(self, func: Callable[[V0], V1], n_worker: int = 0):
        self.func = func
        if n_worker != 0:
            self.pool = multiprocessing.Pool(n_worker)
        else:
            self.pool = None

    @logger.function_block("Map.__call__")
    def __call__(self, values: List[V0]) -> List[V1]:
        if self.pool is None:
            return [self.func(v0)
                    for v0 in logger.iterable_block("values", values)]
        else:
            return list(self.pool.map(self.func, values))


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

    def __call__(self, entry: Environment) -> Optional[Any]:
        return entry[self.key] if self.key in entry.to_dict() else None


def share_memory(model: nn.Module):
    for k, v in model.state_dict().items():
        v.share_memory_()
    return model
