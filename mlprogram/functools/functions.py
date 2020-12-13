import functools
import traceback
from collections import OrderedDict
from typing import Any, Callable, Generic, List, Optional, TypeVar

from torch import multiprocessing

from mlprogram import logging

logger = logging.Logger(__name__)

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
    def __init__(self, func: Callable[[V0], V1], n_worker: int = 0):
        self.func = func
        if n_worker != 0:
            self.pool = multiprocessing.Pool(n_worker)
        else:
            self.pool = None

    @staticmethod
    def _apply(func: Callable[[V0], V1], value: V0) -> Optional[V1]:
        try:
            return func(value)
        except Exception as e:  # noqa
            logger.error(traceback.format_exc())
            return None

    @logger.function_block("Map.__call__")
    def __call__(self, values: List[V0]) -> List[Optional[V1]]:
        if self.pool is None:
            return [Map._apply(self.func, v0)
                    for v0 in logger.iterable_block("values", values)]
        else:
            return list(self.pool.map(
                functools.partial(Map._apply, self.func),
                values))


class Identity(object):
    def __call__(self, x):
        return x
