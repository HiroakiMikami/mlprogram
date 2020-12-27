import logging
import sys
import time
from contextlib import contextmanager
from typing import Callable, Dict, Iterable, Optional, Tuple

import torch
from pytorch_pfn_extras.reporting import report
from torch.autograd.profiler import record_function

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
        self.elapsed_time_log: Dict[str, Tuple[int, float]] = {}
        self.gpu_elapsed_time_log: Dict[str, Tuple[int, float]] = {}

    def dump_elapsed_time_log(self) -> None:
        report({
            f"time.{tag}": time / n
            for tag, (n, time) in self.elapsed_time_log.items()
        })
        report({
            f"gpu.time.{tag}": time / n
            for tag, (n, time) in self.gpu_elapsed_time_log.items()
        })
        self.elapsed_time_log = {}
        self.gpu_elapsed_time_log = {}

    @contextmanager
    def block(self, tag: str, time_tag: Optional[str] = None,
              monitor_gpu_utils: bool = False):
        self.debug(f"start {tag}")
        time_tag = time_tag or tag
        if not torch.cuda.is_available():
            monitor_gpu_utils = False
        if monitor_gpu_utils:
            torch.cuda.synchronize()
        begin = time.time()
        try:
            with record_function(f"{self.name}:{tag}"):
                yield
        finally:
            self.debug(f"finish {tag}")
            if time_tag not in self.elapsed_time_log:
                self.elapsed_time_log[time_tag] = (0, 0.0)
            if monitor_gpu_utils and time_tag not in self.gpu_elapsed_time_log:
                self.gpu_elapsed_time_log[time_tag] = (0, 0.0)
            elapsed_time = time.time() - begin
            n, t = self.elapsed_time_log[time_tag]
            self.elapsed_time_log[time_tag] = (n + 1, t + elapsed_time)
            if monitor_gpu_utils:
                torch.cuda.synchronize()
                gpu_elapsed_time = time.time() - begin
                n, t = self.gpu_elapsed_time_log[time_tag]
                self.gpu_elapsed_time_log[time_tag] = \
                    (n + 1, t + gpu_elapsed_time)

    def function_block(self, tag: str,
                       monitor_gpu_utils: bool = False) \
            -> Callable[[Callable], Callable]:
        def wrapper(f):
            def wrapped(*args, **kwargs):
                with self.block(tag, monitor_gpu_utils=monitor_gpu_utils):
                    return f(*args, **kwargs)
            return wrapped
        return wrapper

    def iterable_block(self, tag: str, iter: Iterable,
                       monitor_gpu_utils: bool = False) -> Iterable:
        def wrapped() -> Iterable:
            for i, x in enumerate(iter):
                with self.block(f"{tag}-{i}", tag, monitor_gpu_utils):
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
