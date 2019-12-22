import ast as python_ast
import transpyle
from typing import Union
from nl2prog.language.ast import AST
from .python_ast_to_ast import to_ast
from .ast_to_python_ast import to_python_ast


def parse(code: str) -> Union[AST, None]:
    try:
        return to_ast(python_ast.parse(code).body[0])
    except:  # noqa
        return None


def unparse(ast: AST) -> Union[str, None]:
    unparser = transpyle.python.unparser.NativePythonUnparser()

    try:
        return unparser.unparse(to_python_ast(ast))
    except:  # noqa
        return None
