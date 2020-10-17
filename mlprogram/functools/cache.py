import os
from typing import Callable
from typing import TypeVar
from typing import Generic

from mlprogram import logging
import torch


V = TypeVar("V")
logger = logging.Logger(__name__)


class FileCache(Generic[V]):
    def __init__(self, path: str, f: Callable[[], V]):
        self.path = path
        self.f = f

    def __call__(self) -> V:
        if os.path.exists(self.path):
            logger.info(f"Cached file found in {self.path}")
            return torch.load(self.path)
        else:
            logger.info(f"Cached file not found in {self.path}")
            val = self.f()
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            torch.save(val, self.path)
            return val
        return self.f()


def file_cache(path: str) -> Callable[[Callable[[], V]], FileCache[V]]:
    def wrapper(f: Callable[[], V]):
        return FileCache(path, f)
    return wrapper
