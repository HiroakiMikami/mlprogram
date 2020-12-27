import os
from typing import Callable, Generic, TypeVar, cast

import torch

from mlprogram import logging

V = TypeVar("V")
logger = logging.Logger(__name__)


class FileCache(Generic[V]):
    def __init__(self, path: str, f: Callable[[], V]):
        self.path = path
        self.f = f

    def __call__(self) -> V:
        if os.path.exists(self.path):
            logger.info(f"Cached file found in {self.path}")
            return cast(V, torch.load(self.path))
        else:
            logger.info(f"Cached file not found in {self.path}")
            val = self.f()
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            torch.save(val, self.path)
            return val


def file_cache(path: str) -> Callable[[Callable[[], V]], FileCache[V]]:
    def wrapper(f: Callable[[], V]):
        return FileCache(path, f)
    return wrapper
