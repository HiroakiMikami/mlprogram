from typing import List, Optional, Tuple

from mlprogram import logging
from mlprogram.languages import Token
from mlprogram.languages.c import Lexer as BaseLexer

logger = logging.Logger(__name__)


class _Mapping:
    def __init__(self, prefix):
        self.mapping = {}
        self.prefix = prefix

    def __call__(self, value):
        if value not in self.mapping:
            self.mapping[value] = len(self.mapping)
        return f"___{self.prefix}@{self.mapping[value]}___"


class Lexer(BaseLexer):
    def __init__(self, delimiter: str = " "):
        super().__init__()
        self.delimiter = delimiter

    def tokenize_with_offset(self, code: str) \
            -> Optional[List[Tuple[int, Token[str, str]]]]:
        tokens = super().tokenize_with_offset(code)
        id_to_idx = _Mapping("id")
        int_to_idx = _Mapping("int")
        float_to_idx = _Mapping("float")
        str_to_idx = _Mapping("string")
        chr_to_idx = _Mapping("char")

        if tokens is None:
            return None
        retval = []
        for offset, token in tokens:
            if token.kind == "ID":
                retval.append((offset, Token(token.kind, id_to_idx(token.raw_value),
                                             token.raw_value)))
            elif token.kind is not None and token.kind.startswith("INT_CONST_"):
                retval.append((offset, Token(token.kind, int_to_idx(token.raw_value),
                                             token.raw_value)))
            elif token.kind == "FLOAT_CONST":
                retval.append((offset, Token(token.kind, float_to_idx(token.raw_value),
                                             token.raw_value)))
            elif token.kind == "STRING_LITERAL":
                retval.append((offset, Token(token.kind, str_to_idx(token.raw_value),
                                             token.raw_value)))
            elif token.kind == "CHAR_CONST":
                retval.append((offset, Token(token.kind, chr_to_idx(token.raw_value),
                                             token.raw_value)))
            else:
                retval.append((offset, token))
        return retval
