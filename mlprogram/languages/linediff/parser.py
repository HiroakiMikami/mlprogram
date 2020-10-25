from typing import Optional
from typing import TypeVar
from typing import Generic
from typing import List
from typing import cast
from mlprogram.languages.linediff import AST as diffAST
from mlprogram.languages.linediff import Diff
from mlprogram.languages.linediff import Delta
from mlprogram.languages.linediff import Insert
from mlprogram.languages.linediff import Remove
from mlprogram.languages.linediff import Replace
from mlprogram.languages import Lexer
from mlprogram.languages import AST
from mlprogram.languages import Node
from mlprogram.languages import Leaf
from mlprogram.languages import TokenSequence
from mlprogram.languages import Token
from mlprogram.languages import Sugar as S
from mlprogram.languages import Parser as BaseParser
from mlprogram import logging

logger = logging.Logger(__name__)

Kind = TypeVar("Kind")


class Parser(BaseParser[diffAST], Generic[Kind]):
    def __init__(self, lexer: Lexer[Kind]):
        super().__init__()
        self.lexer = lexer

    def parse(self, code: diffAST) -> Optional[AST]:
        if isinstance(code, Diff):
            deltas = [self.parse(delta) for delta in code.deltas]
            if None in deltas:
                return None
            return S.node(
                "Diff",
                deltas=("Delta", deltas))
        elif isinstance(code, Insert):
            token_sequence = self.lexer.tokenize(code.value)
            if token_sequence is None:
                return None
            return S.node(
                "Insert",
                line_number=("int", S.leaf("int", code.line_number)),
                value=("str", [S.leaf("str", token.value)
                               for _, token in token_sequence.tokens])
            )
        elif isinstance(code, Remove):
            return S.node(
                "Remove",
                line_number=("int", S.leaf("int", code.line_number)))
        elif isinstance(code, Replace):
            token_sequence = self.lexer.tokenize(code.value)
            if token_sequence is None:
                return None
            return S.node(
                "Replace",
                line_number=("int", S.leaf("int", code.line_number)),
                value=("str", [S.leaf("str", token.value)
                               for _, token in token_sequence.tokens])
            )
        logger.warning(f"Invalid node type {code.get_type_name()}")
        # TODO throw exception
        return None

    def unparse(self, code: AST) -> Optional[diffAST]:
        assert isinstance(code, Node)
        fields = {field.name: field.value for field in code.fields}
        if code.get_type_name() == "Diff":
            deltas = [self.unparse(delta)
                      for delta in cast(List[AST], fields["deltas"])]
            if None in deltas:
                return None
            return Diff(cast(List[Delta], deltas))
        elif code.get_type_name() == "Insert":
            # TODO should not instantiate token sequence
            value = self.lexer.untokenize(TokenSequence([
                (-1, Token(None, cast(Leaf, v).value, cast(Leaf, v).value))
                for v in cast(List[AST], fields["value"])
            ]))
            if value is None:
                return None
            return Insert(cast(Leaf, fields["line_number"]).value, value)
        elif code.get_type_name() == "Remove":
            return Remove(cast(Leaf, fields["line_number"]).value)
        elif code.get_type_name() == "Replace":
            # TODO should not instantiate token sequence
            value = self.lexer.untokenize(TokenSequence([
                (-1, Token(None, cast(Leaf, v).value, cast(Leaf, v).value))
                for v in cast(List[AST], fields["value"])
            ]))
            if value is None:
                return None
            return Replace(cast(Leaf, fields["line_number"]).value, value)
        raise AssertionError(f"invalid node type_name: {code.get_type_name()}")
