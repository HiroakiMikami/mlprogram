from typing import Generic, Optional, TypeVar

from mlprogram.languages.token_sequence import TokenSequence

Kind = TypeVar("Kind")


class Lexer(Generic[Kind]):
    def tokenize(self, text: str) -> Optional[TokenSequence[Kind]]:
        raise NotImplementedError

    def untokenize(self, sequnece: TokenSequence[Kind]) -> Optional[str]:
        raise NotImplementedError
