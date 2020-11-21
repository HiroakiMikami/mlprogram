from typing import List, Optional, Tuple

from pycparser.c_lexer import CLexer
from pycparser.ply.lex import LexToken

from mlprogram import logging
from mlprogram.languages import Lexer as BaseLexer
from mlprogram.languages import Token

logger = logging.Logger(__name__)


class Lexer(BaseLexer[str, str]):
    def __init__(self, delimiter: str = " "):
        super().__init__()
        self.delimiter = delimiter

    def tokenize_with_offset(self, code: str) \
            -> Optional[List[Tuple[int, Token[str, str]]]]:
        code = code.replace("\r", "")

        lexer = CLexer(logger.warning, lambda: None,
                       lambda: None, lambda x: False)
        lexer.build(optimize=False)
        lexer.input(code)
        tokens: List[LexToken] = list(iter(lexer.token, None))

        return [
            (token.lexpos, Token(token.type, token.value, token.value))
            for token in tokens
        ]

    def untokenize(self, sequence: List[Token[str, str]]) -> Optional[str]:
        return self.delimiter.join(token.raw_value
                                   for token in sequence)
