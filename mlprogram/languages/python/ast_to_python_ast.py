import ast as python_ast  # noqa
# from typing import List

import mlprogram.asts as ast
from mlprogram.languages.python import PythonAST
from .utils import BuiltinType


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
    if isinstance(target, ast.Node):
        # Node
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
                for child in field.value:
                    assert isinstance(child, ast.Node)
                    assert isinstance(field.type_name, str)
                    if field.type_name.endswith("__list"):
                        if isinstance(child.fields[0].value, list):
                            for ch in child.fields[0].value:
                                elems.append(to_python_ast(ch))
                        else:
                            elems.append(to_python_ast(child.fields[0].value))
                    else:
                        elems.append(to_python_ast(child))
                setattr(node, name, elems)
            else:
                setattr(node, name, to_python_ast(field.value))

        return node
    elif isinstance(target, ast.Leaf):
        # Leaf
        assert isinstance(target.type_name, str)
        return to_builtin_type(target.value, target.type_name)
    else:
        raise Exception(f"Invalid arugment: {target} is not AST")
