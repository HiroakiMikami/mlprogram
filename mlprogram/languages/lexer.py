from typing import Generic, List, Optional, TypeVar

from mlprogram.languages.token import Token

Kind = TypeVar("Kind")
Value = TypeVar("Value")


class Lexer(Generic[Kind, Value]):
    def tokenize(self, text: str) -> Optional[List[Token[Kind, Value]]]:
        raise NotImplementedError

    def untokenize(self, sequnece: List[Token[Kind, Value]]) -> Optional[str]:
        raise NotImplementedError
