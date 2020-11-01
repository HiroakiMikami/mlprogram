from dataclasses import dataclass
from typing import Any, Generic, List, Tuple, TypeVar

from mlprogram.languages.token import Token

Kind = TypeVar("Kind")


@dataclass
class TokenSequence(Generic[Kind]):
    tokens: List[Tuple[int, Token[Kind, str]]]

    def __hash__(self) -> int:
        return hash(tuple(self.tokens))

    def __eq__(self, rhs: Any) -> bool:
        if not isinstance(rhs, TokenSequence):
            return False
        return self.tokens == rhs.tokens

    def __str__(self) -> str:
        return ",".join(map(str, self.tokens))
