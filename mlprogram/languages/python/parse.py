import ast as python_ast
import transpyle
from enum import Enum
from typing import Optional
from mlprogram.asts import AST
from .python_ast_to_ast import to_ast
from .ast_to_python_ast import to_python_ast


class ParseMode(Enum):
    Single = 1
    Eval = 2
    Exec = 3


class Parse:
    def __init__(self, mode: ParseMode = ParseMode.Single):
        self.mode = mode

    def __call__(self, code: str) -> Optional[AST]:
        try:
            past = python_ast.parse(code)
            if self.mode == ParseMode.Exec:
                return to_ast(past)
            else:
                return to_ast(past.body[0])
        except:  # noqa
            return None


class Unparse:
    def __call__(self, ast: AST) -> Optional[str]:
        unparser = transpyle.python.unparser.NativePythonUnparser()

        try:
            return unparser.unparse(to_python_ast(ast))
        except:  # noqa
            return None
