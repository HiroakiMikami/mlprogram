from typing import List, Optional, Tuple

from pycparser.c_lexer import CLexer
from pycparser.ply.lex import LexToken

from mlprogram import logging
from mlprogram.languages import Lexer as BaseLexer
from mlprogram.languages import Token, TokenSequence

logger = logging.Logger(__name__)


class Lexer(BaseLexer[str]):
    def __init__(self, delimiter: str = " "):
        super().__init__()
        self.delimiter = delimiter

    def tokenize(self, code: str) -> Optional[TokenSequence]:
        lines = list(code.split("\n"))
        offsets = [0]
        for line in lines:
            offsets.append(offsets[-1] + len(line) + 1)

        lexer = CLexer(logger.warning, lambda: None,
                       lambda: None, lambda x: False)
        lexer.build(optimize=False)
        lexer.input(code)
        tokens: List[LexToken] = list(iter(lexer.token, None))

        tokens_with_offset: List[Tuple[int, Token[str, str]]] = [
            (token.lexpos, Token(token.type, token.value, token.value))
            for token in tokens
        ]

        return TokenSequence(tokens_with_offset)

    def untokenize(self, sequence: TokenSequence) -> Optional[str]:
        return self.delimiter.join(token[1].raw_value
                                   for token in sequence.tokens)
