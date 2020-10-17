from typing import Generic, TypeVar, Optional, Any, List, Callable
from mlprogram import Environment
from mlprogram import logging

logger = logging.Logger(__name__)

V = TypeVar("V")


class Identity(Generic[V]):
    def __call__(self, value: V) -> V:
        return value


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
