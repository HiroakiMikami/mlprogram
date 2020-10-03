from dataclasses import dataclass
from typing import Any
from typing import TypeVar
from typing import Generic
from typing import Tuple
from typing import List
from mlprogram.languages.token import Token


Kind = TypeVar("Kind")


@dataclass
class TokenSequence(Generic[Kind]):
    tokens: List[Tuple[int, Token[Kind, str]]]
    raw_text: str

    def __hash__(self) -> int:
        return hash((tuple(self.tokens), self.raw_text))

    def __eq__(self, rhs: Any) -> bool:
        if not isinstance(rhs, TokenSequence):
            return False
        return self.tokens == rhs.tokens and self.raw_text == rhs.raw_text

    def __str__(self) -> str:
        return ",".join(map(str, self.tokens))
