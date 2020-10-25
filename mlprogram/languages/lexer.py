from typing import TypeVar
from typing import Generic
from typing import Optional

from mlprogram.languages import TokenSequence


Kind = TypeVar("Kind")


class Lexer(Generic[Kind]):
    def tokenize(self, text: str) -> Optional[TokenSequence[Kind]]:
        raise NotImplementedError

    def untokenize(self, sequnece: TokenSequence[Kind]) -> Optional[str]:
        raise NotImplementedError
