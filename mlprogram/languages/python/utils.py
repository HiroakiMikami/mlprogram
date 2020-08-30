import ast as python_ast
from typing import Union
from mlprogram.asts import Root

BuiltinType = Union[int, float, bool, str, bytes, object, None]
PythonAST = Union[python_ast.AST, BuiltinType]


def is_builtin_type(node: PythonAST) -> bool:
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


class IsSubtype:
    def __call__(self, subtype: Union[str, Root],
                 basetype: Union[str, Root]) -> bool:
        if isinstance(basetype, Root):
            return True
        if isinstance(subtype, Root):
            return False
        if subtype.endswith("__proxy") or \
                basetype.endswith("__proxy"):
            return subtype == basetype
        try:
            sub = eval(f"python_ast.{subtype}()")
        except:  # noqa
            sub = eval(f"{subtype}()")
        try:
            base = eval(f"python_ast.{basetype}()")
        except:  # noqa
            base = eval(f"{basetype}()")
        return isinstance(sub, type(base))
