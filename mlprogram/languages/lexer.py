from typing import Generic, List, Optional, Tuple, TypeVar

from mlprogram.languages.token import Token

Kind = TypeVar("Kind")
Value = TypeVar("Value")


class Lexer(Generic[Kind, Value]):
    def tokenize_with_offset(self, text: str) \
            -> Optional[List[Tuple[int, Token[Kind, Value]]]]:
        raise NotImplementedError

    def tokenize(self, text: str) -> Optional[List[Token[Kind, Value]]]:
        tokens = self.tokenize_with_offset(text)
        if tokens is None:
            return None
        return [token for _, token in tokens]

    def untokenize(self, sequnece: List[Token[Kind, Value]]) -> Optional[str]:
        raise NotImplementedError
