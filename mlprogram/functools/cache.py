import os
import uuid
from typing import Callable, Generic, TypeVar, cast

import torch

from mlprogram import distributed, logging

V = TypeVar("V")
logger = logging.Logger(__name__)


class FileCache(Generic[V]):
    def __init__(self, path: str, f: Callable[[], V]):
        self.path = path
        self.f = f

    def __call__(self) -> V:
        if distributed.is_main_process():
            if not os.path.exists(self.path):
                logger.info(f"Cached file not found in {self.path}")
                tmpfile = os.path.join(os.path.dirname(self.path), str(uuid.uuid4()))
                val = self.f()
                os.makedirs(os.path.dirname(self.path), exist_ok=True)
                torch.save(val, tmpfile)
                os.rename(tmpfile, self.path)
        distributed.call(torch.distributed.barrier)
        logger.info(f"Cached file found in {self.path}")
        return cast(V, torch.load(self.path))


def file_cache(path: str) -> Callable[[Callable[[], V]], FileCache[V]]:
    def wrapper(f: Callable[[], V]):
        return FileCache(path, f)
    return wrapper


def with_file_cache(path: str, f: Callable, *args, **kwargs):
    @file_cache(path)
    def g():
        return f(*args, **kwargs)
    return g()
