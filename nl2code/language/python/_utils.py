import ast as python_ast
from typing import Union
from nl2code.language.action import NodeType

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


def is_subtype(subtype: NodeType, basetype: NodeType) -> bool:
    if subtype.type_name.endswith("__list") or \
            basetype.type_name.endswith("__list"):
        return subtype == basetype
    subtype = subtype.type_name
    basetype = basetype.type_name
    try:
        sub = eval("python_ast.{}()".format(subtype))
    except:  # noqa
        sub = eval("{}()".format(subtype))
    try:
        base = eval("python_ast.{}()".format(basetype))
    except:  # noqa
        base = eval("{}()".format(basetype))
    return isinstance(sub, type(base))
