from dataclasses import dataclass
from typing import List, Generic, TypeVar, Optional, Any


V = TypeVar("V")


@dataclass
class Token(Generic[V]):
    type_name: Optional[str]
    value: V

    def __hash__(self) -> int:
        return hash(self.type_name) ^ hash(self.value)

    def __eq__(self, other):
        if isinstance(other, Token):
            return self.type_name == other.type_name and \
                    self.value == other.value
        else:
            return False


@dataclass
class Query:
    reference: List[Token[str]]
    query_for_dnn: List[str]


@dataclass
class Reference:
    name: str

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Reference):
            return self.name == other.name
        else:
            return False
