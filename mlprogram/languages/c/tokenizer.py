from dataclasses import dataclass
from typing import List
from pycparser.c_lexer import CLexer
from pycparser.ply.lex import LexToken
from mlprogram import logging

logger = logging.Logger(__name__)


@dataclass
class Token:
    value: str  # TODO typevar?
    offset: int
    kind: str  # TODO typevar?


class Tokenizer(object):
    def __call__(self, code: str) -> List[Token]:
        lines = list(code.split("\n"))
        offsets = [0]
        for line in lines:
            offsets.append(offsets[-1] + len(line) + 1)

        lexer = CLexer(logger.warning, lambda: None,
                       lambda: None, lambda x: False)
        lexer.build(optimize=False)
        lexer.input(code)
        tokens: List[LexToken] = list(iter(lexer.token, None))
        return [Token(token.value, token.lexpos,
                      token.type)
                for token in tokens]
