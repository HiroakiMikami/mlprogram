import ast as python_ast
import transpyle
from enum import Enum
from typing import Optional
from mlprogram.ast import AST
from .python_ast_to_ast import to_ast
from .ast_to_python_ast import to_python_ast


class ParseMode(Enum):
    Single = 1
    Eval = 2
    Exec = 3


def parse(code: str, mode: ParseMode = ParseMode.Single) -> Optional[AST]:
    try:
        past = python_ast.parse(code)
        if mode == ParseMode.Exec:
            return to_ast(past)
        else:
            return to_ast(past.body[0])
    except:  # noqa
        return None


def unparse(ast: AST) -> Optional[str]:
    unparser = transpyle.python.unparser.NativePythonUnparser()

    try:
        return unparser.unparse(to_python_ast(ast))
    except:  # noqa
        return None
