import ast as python_ast
from typing import Union

BuiltinType = Union[int, float, bool, str, bytes, object, type(None)]
PythonAST = Union[python_ast.AST, BuiltinType]


def is_builtin_type(node):
    if isinstance(node, python_ast.AST):
        return False
    elif isinstance(node, int):
        return True
    elif isinstance(node, float):
        return True
    elif isinstance(node, bool):
        return True
    elif isinstance(node, str):
        return True
    elif isinstance(node, bytes):
        return True
    elif isinstance(node, object):
        return True
    return False
