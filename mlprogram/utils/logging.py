import logging
import numpy as np
from typing import Callable, Iterable, Dict, Optional, List
from contextlib import contextmanager
from torch.autograd.profiler import record_function
from pytorch_pfn_extras.reporting import report
import time
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
        self.elapsed_time_log: Dict[str, List[float]] = {}

    def dump_eplased_time_log(self) -> None:
        report({
            f"time.{tag}": np.mean(time)
            for tag, time in self.elapsed_time_log.items()
        })
        self.elapsed_time_log = {}

    @contextmanager
    def block(self, tag: str, time_tag: Optional[str] = None):
        self.debug(f"start {tag}")
        time_tag = time_tag or tag
        begin = time.time()
        try:
            with record_function(f"{self.name}:{tag}"):
                yield
        finally:
            self.debug(f"finish {tag}")
            if time_tag not in self.elapsed_time_log:
                self.elapsed_time_log[time_tag] = []
            self.elapsed_time_log[time_tag].append(time.time() - begin)

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
                with self.block(f"{tag}-{i}", tag):
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
