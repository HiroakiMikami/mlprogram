import ast as python_ast  # noqa
# from typing import List

import mlprogram.languages.ast as ast
from mlprogram.languages.python import PythonAST
from mlprogram.languages.python.utils import BuiltinType


def to_builtin_type(value: str, type_name: str) -> BuiltinType:
    if type_name == "bytes":
        return str(value).encode()
    elif type_name == "NoneType":
        return None
    elif type_name == "str":
        return value
    else:
        try:
            return eval(f"{type_name}({value})")
        except Exception:
            return value


def to_python_ast(target: ast.AST) -> PythonAST:
    assert isinstance(target, ast.Node)
    if str(target.type_name).endswith("__proxy"):
        assert isinstance(target.fields[0].value, list)
        value = ""
        tpe = str(target.type_name).replace("__proxy", "")
        for child in target.fields[0].value:
            assert isinstance(child, ast.Leaf)
            value += child.value
        return to_builtin_type(value, tpe)

    type_name = target.type_name
    node = eval(f"python_ast.{type_name}()")

    # Fill all fields
    for field_name in node._fields:
        setattr(node, field_name, None)

    for field in target.fields:
        name = field.name
        if isinstance(field.value, list):
            # List
            elems = []
            if len(field.value) > 0 and \
                    isinstance(field.value[0], ast.Leaf):
                # Leaf
                value = ""
                for child in field.value:
                    assert isinstance(child, ast.Leaf)
                    value += child.value
                setattr(node, name,
                        to_builtin_type(value, str(field.type_name)))
            else:
                # Node
                for child in field.value:
                    assert isinstance(child, ast.Node)
                    assert isinstance(field.type_name, str)
                    elems.append(to_python_ast(child))
                setattr(node, name, elems)
        else:
            setattr(node, name, to_python_ast(field.value))

    return node
