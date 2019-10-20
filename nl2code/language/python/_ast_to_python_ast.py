import ast as python_ast # noqa
# from typing import List

import nl2code.language.ast as ast
from nl2code.language.python import PythonAST
from ._utils import BuiltinType


def to_builtin_type(value: str, type_name: str) -> BuiltinType:
    if type_name == "bytes":
        return str(value).encode()
    elif type_name == "NoneType":
        return None
    else:
        return eval("{}(\"{}\")".format(type_name, value))


def to_python_ast(target: ast.AST) -> PythonAST:
    if isinstance(target, ast.Node):
        # Node
        type_name = target.type_name.replace("__list", "")
        node = eval("python_ast.{}()".format(type_name))

        # Fill all fields
        for field_name in node._fields:
            setattr(node, field_name, None)

        def unwrap_cast(field: ast.Field):
            if isinstance(field.value, ast.Leaf):
                return field.value
            if len(field.value.fields) != 1:
                return field.value
            f = field.value.fields[0]
            if f.name == "__cast":
                return f.value
            return field.value

        for field in target.fields:
            name = field.name
            if field.type_name.endswith("__list"):
                # List
                child: ast.Node = field.value
                elems = []
                for f in child.fields:
                    elems.append(to_python_ast(unwrap_cast(f)))
                setattr(node, name, elems)
            else:
                setattr(node, name, to_python_ast(unwrap_cast(field)))

        return node
    else:
        # Leaf
        return to_builtin_type(target.value, target.type_name)
