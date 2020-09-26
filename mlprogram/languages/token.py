from dataclasses import dataclass
from typing import Any
from typing import TypeVar
from typing import Generic
from typing import Optional


Kind = TypeVar("Kind")
Value = TypeVar("Value")


@dataclass
class Token(Generic[Kind, Value]):
    kind: Optional[Kind]
    value: Value
    raw_value: Value

    def __hash__(self) -> int:
        return hash((self.kind, self.value, self.raw_value))

    def __eq__(self, rhs: Any) -> bool:
        if not isinstance(rhs, Token):
            return False
        return self.kind == rhs.kind and self.value == rhs.value and \
            self.raw_value == rhs.raw_value
