import argparse
from typing import Any, Dict, List, Type

from mlprogram.logging import Logger

logger = Logger(__name__)


class Options:
    _values: Dict[str, Any]

    def __init__(self):
        self._values = {}

    def __setitem__(self, key: str, value: Any):
        if key in self._values:
            logger.info(f"option {key} is already defined with the value of {value}")
            return
        self._values[key] = value

    def overwrite_option(self, key: str, value: Any):
        assert key in self._values
        logger.info(f"Overwrite option {key}: {self._values[key]} -> {value}")
        self._values[key] = value

    def __getattr__(self, name: str) -> Any:
        return self._values[name]

    @property
    def options(self) -> Dict[str, Type]:
        return {k: type(v) for k, v in self._values.items()}

    def overwrite_options_by_args(self, args: List[str]) -> None:
        parser = self._parser()
        ret = parser.parse_args(args)
        for k, v in self._values.items():
            self._values[k] = getattr(ret, k, v)

    def _parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()
        for k, v in self._values.items():
            parser.add_argument(f"--{k}", type=type(v), default=v)
        return parser


global_options = Options()
