import ast as python_ast
from typing import Union
from mlprogram.asts import Root
import re
from nltk import tokenize
from typing import List

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
        if subtype.endswith("__list") or \
                basetype.endswith("__list"):
            return subtype == basetype
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


tokenizer = tokenize.WhitespaceTokenizer()


class TokenizeToken:
    def __init__(self, split_camel_case: bool = False):
        self.split_camel_case = split_camel_case

    def __call__(self, value: str) -> List[str]:
        if self.split_camel_case and re.search(
                r"^[A-Z].*", value) and (" " not in value):
            # Camel Case
            words = re.findall(r"[A-Z][a-z]+", value)
            if "".join(words) == value:
                return words
            else:
                return [value]
        else:
            # Divide by space
            words = re.split(r"( +)", value)
            return [word for word in words if word != ""]
