from typing import Any, Optional

from mlprogram.builtins.datatypes import Environment


class Pick(object):
    def __init__(self, key: str):
        self.key = key

    def __call__(self, entry: Environment) -> Optional[Any]:
        return entry[self.key] if self.key in entry.to_dict() else None
