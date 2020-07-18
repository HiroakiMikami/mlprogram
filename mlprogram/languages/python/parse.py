import ast as python_ast
import transpyle
from enum import Enum
from typing import Optional, Callable, List
from mlprogram.asts import AST
from .python_ast_to_ast import to_ast
from .ast_to_python_ast import to_python_ast


class ParseMode(Enum):
    Single = 1
    Eval = 2
    Exec = 3


class Parse:
    def __init__(self, tokenize: Optional[Callable[[str], List[str]]] = None,
                 retain_variadic_fields: bool = True,
                 mode: ParseMode = ParseMode.Single):
        self.mode = mode
        self.tokenize = tokenize
        self.retain_variadic_fields = retain_variadic_fields

    def __call__(self, code: str) -> Optional[AST]:
        try:
            past = python_ast.parse(code)
            if self.mode == ParseMode.Exec:
                print("test", code)
                return to_ast(
                    past,
                    tokenize=self.tokenize,
                    retain_variadic_fields=self.retain_variadic_fields)
            else:
                return to_ast(
                    past.body[0],
                    tokenize=self.tokenize,
                    retain_variadic_fields=self.retain_variadic_fields)
        except:  # noqa
            return None


class Unparse:
    def __call__(self, ast: AST) -> Optional[str]:
        unparser = transpyle.python.unparser.NativePythonUnparser()

        try:
            return unparser.unparse(to_python_ast(ast))
        except:  # noqa
            return None
