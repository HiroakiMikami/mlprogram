import ast as python_ast
from functools import lru_cache
from typing import Union

from mlprogram import logging
from mlprogram.languages import Root

logger = logging.Logger(__name__)

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


@lru_cache(maxsize=1000)
def to_python_variable(name: str):
    try:
        return eval(f"python_ast.{name}()")
    except:  # noqa
        return eval(f"{name}()")


class IsSubtype:
    @logger.function_block("IsSubType.__call__")
    def __call__(self, subtype: Union[str, Root],
                 basetype: Union[str, Root]) -> bool:
        if isinstance(basetype, Root):
            return True
        if isinstance(subtype, Root):
            return False
        if subtype.endswith("__proxy") or \
                basetype.endswith("__proxy"):
            return subtype == basetype
        sub = to_python_variable(subtype)
        base = to_python_variable(basetype)
        return isinstance(sub, type(base))
