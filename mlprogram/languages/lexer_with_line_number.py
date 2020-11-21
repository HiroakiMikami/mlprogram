from typing import Generic, List, Optional, Tuple, TypeVar, Union, cast

from mlprogram.languages.kinds import Kinds
from mlprogram.languages.lexer import Lexer
from mlprogram.languages.token import Token

Kind = TypeVar("Kind")
Value = TypeVar("Value")


class LexerWithLineNumber(Lexer[Union[Kinds.LineNumber, Kind], Union[int, Value]],
                          Generic[Kind, Value]):
    def __init__(self, lexer: Lexer[Kind, Value]):
        self.lexer = lexer

    def tokenize_with_offset(self, text: str) \
            -> Optional[List[Tuple[int, Token[Union[Kinds.LineNumber, Kind],
                                              Union[int, Value]]]]]:
        lines = text.split("\n")
        offsets = [0]
        for line in lines:
            offsets.append(offsets[-1] + len(line) + 1)
        tokens: List[Tuple[int, Token[Union[Kinds.LineNumber, Kind],
                                      Union[int, Value]]]] = []
        linenum = 0
        origs = self.lexer.tokenize_with_offset(text)
        if origs is None:
            return None
        for offset, token in origs:
            if offsets[linenum] <= offset:
                tokens.append((offset, Token(Kinds.LineNumber(), linenum, linenum)))
                linenum += 1
            tokens.append((offset,
                           cast(Token[Union[Kinds.LineNumber, Kind], Union[int, Value]],
                                token)))
        return tokens

    def untokenize(self, sequnece: List[Token[Union[Kinds.LineNumber, Kind],
                                              Union[int, Value]]]
                   ) -> Optional[str]:
        line: List[Token[Kind, Value]] = []
        text = ""
        for token in sequnece:
            if token.kind == Kinds.LineNumber():
                if len(line) != 0:
                    linestr = self.lexer.untokenize(line)
                    if linestr is None:
                        return None
                    text += linestr
                    text += "\n"
                line = []
            else:
                line.append(cast(Token[Kind, Value], token))
        if len(line) != 0:
            linestr = self.lexer.untokenize(line)
            if linestr is None:
                return None
            text += linestr
            text += "\n"

        return text
