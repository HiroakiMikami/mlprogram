import logging
from typing import Callable, Iterable
from contextlib import contextmanager
from torch.autograd.profiler import record_function
import sys
from mlprogram import transpyle  # noqa


def set_level(level):
    if sys.version_info[:2] >= (3, 8):
        # Python 3.8 or later
        logging.basicConfig(level=level, stream=sys.stdout,
                            force=True)
    else:
        logging.root.handlers[0].setLevel(level)


class Logger(object):
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(name)

    @contextmanager
    def block(self, tag: str):
        self.debug(f"start {tag}")
        try:
            with record_function(f"{self.name}:{tag}"):
                yield
        finally:
            self.debug(f"finish {tag}")

    def function_block(self, tag: str) -> Callable[[Callable], Callable]:
        def wrapper(f):
            def wrapped(*args, **kwargs):
                with self.block(tag):
                    return f(*args, **kwargs)
            return wrapped
        return wrapper

    def iterable_block(self, tag: str, iter: Iterable) -> Iterable:
        def wrapped():
            for i, x in enumerate(iter):
                with self.block(f"{tag}-{i}"):
                    yield x
        return wrapped()

    def log(self, *args, **kwargs) -> None:
        self.logger.log(*args, **kwargs)

    def info(self, *args, **kwargs) -> None:
        self.logger.info(*args, **kwargs)

    def debug(self, *args, **kwargs) -> None:
        self.logger.debug(*args, **kwargs)

    def warning(self, *args, **kwargs) -> None:
        self.logger.warning(*args, **kwargs)

    def error(self, *args, **kwargs) -> None:
        self.logger.error(*args, **kwargs)

    def critical(self, *args, **kwargs) -> None:
        self.logger.critical(*args, **kwargs)
