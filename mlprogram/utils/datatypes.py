from dataclasses import dataclass
from typing import List, Generic, TypeVar, Optional


V = TypeVar("V")


@dataclass
class Token(Generic[V]):
    type_name: Optional[str]
    value: V


@dataclass
class Query:
    reference: List[Token[str]]
    query_for_dnn: List[str]
