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


def is_subtype(subtype: str, basetype: str) -> bool:
    if subtype.endswith("__list") or \
            basetype.endswith("__list"):
        return subtype == basetype
    try:
        sub = eval("python_ast.{}()".format(subtype))
    except:  # noqa
        sub = eval("{}()".format(subtype))
    try:
        base = eval("python_ast.{}()".format(basetype))
    except:  # noqa
        base = eval("{}()".format(basetype))
    return isinstance(sub, type(base))