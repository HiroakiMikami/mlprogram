from typing import Generic, List, Optional, TypeVar, Union, cast

from mlprogram.languages.kinds import Kinds
from mlprogram.languages.lexer import Lexer
from mlprogram.languages.token import Token

Kind = TypeVar("Kind")
Value = TypeVar("Value")


class LexerWithLineNumber(Lexer[Union[Kinds.LineNumber, Kind], Union[int, Value]],
                          Generic[Kind, Value]):
    def __init__(self, lexer: Lexer[Kind, Value]):
        self.lexer = lexer

    def tokenize(self, text: str) -> Optional[List[Token[Union[Kinds.LineNumber, Kind],
                                                         Union[int, Value]]]]:
        lines = text.split("\n")
        tokens: List[Token[Union[Kinds.LineNumber, Kind], Union[int, Value]]] = []
        for i, line in enumerate(lines):
            tokens.append(Token(Kinds.LineNumber(), i, i))
            line_tokens = self.lexer.tokenize(line)
            if line_tokens is None:
                return None
            tokens.extend(cast(List[Token[Union[Kinds.LineNumber, Kind],
                                          Union[int, Value]]],
                               line_tokens))
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
