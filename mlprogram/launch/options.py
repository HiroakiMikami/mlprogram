import argparse
from typing import Any, Callable, Dict, List, Optional, Type

from mlprogram.logging import Logger

logger = Logger(__name__)


class Options:
    _values: Dict[str, Any]
    _cached_values: Dict[str, Any]

    def __init__(self):
        self._values = {}
        self._cached_values = {}
        self._hooks = []
        self._args = []

    def __setitem__(self, key: str, value: Any):
        if key in self._values:
            logger.info(f"option {key} is already defined with the value of {value}")
            return
        self._values[key] = value

    def __getattr__(self, name: str) -> Any:
        if name in self._cached_values:
            logger.debug(f"Option {name} is already accessed")
            return self._cached_values[name]

        parser = self._parser()
        ret, _ = parser.parse_known_args(self._args)
        if hasattr(ret, name) and getattr(ret, name) is not None:
            logger.info(f"Overwrite option {name} by arguments")
            ret = getattr(ret, name)
            self._cached_values[name] = ret
            return ret

        v = self._values[name]
        for hook in self._hooks:
            v_opt = hook(name, type(v))
            if v_opt is not None:
                logger.info(f"Overwrite option {name} by hook")
                self._cached_values[name] = v_opt
                return v_opt

        self._cached_values[name] = v
        return v

    def set_hook(self, hook: Callable[[str, Type], Optional[Any]]) -> None:
        self._hooks.append(hook)

    def set_args(self, args: List[str]) -> None:
        self._args = args

    @property
    def options(self) -> Dict[str, Type]:
        return {k: type(v) for k, v in self._values.items()}

    def _parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()
        for k, v in self._values.items():
            parser.add_argument(f"--{k}", type=type(v))
        return parser


global_options = Options()
