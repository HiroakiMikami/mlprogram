from typing import Any

from mlprogram.builtins.datatypes import Environment


class Pack(object):
    def __init__(self, key: str, is_supervision: bool = False):
        self._key = key
        self._is_supervision = is_supervision

    def __call__(self, value: Any) -> Environment:
        out = Environment(
            {self._key: value}
        )
        if self._is_supervision:
            out.mark_as_supervision(self._key)
        return out
