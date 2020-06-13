import ast as python_ast
from typing import List, Type, Union

import mlprogram.asts as ast
from mlprogram.languages.python import PythonAST
from .utils import is_builtin_type, BuiltinType


def base_ast_type(node: PythonAST) \
        -> Union[Type[python_ast.AST], Type[BuiltinType]]:
    """
    Return the base type of the AST
    """
    base_types = set([
        python_ast.mod, python_ast.stmt, python_ast.expr,
        python_ast.expr_context, python_ast.slice, python_ast.boolop,
        python_ast.operator, python_ast.unaryop, python_ast.cmpop,
        python_ast.comprehension, python_ast.excepthandler,
        python_ast.arguments, python_ast.arg, python_ast.keyword,
        python_ast.alias, python_ast.withitem
    ])
    for base in base_types:
        if isinstance(node, base):
            return base
    return type(node)


def to_ast(target: PythonAST) -> ast.AST:
    """
    Return the AST corresponding to the Python AST

    Parameters
    ----------
    target: PythonAST
        The Python AST to be converted

    Returns
    -------
    ast.AST
        The corresponding AST
    """
    type_name = type(target).__name__
    fields: List[ast.Field] = []

    if is_builtin_type(target):
        # Builtin-type
        if isinstance(target, bytes):
            return ast.Leaf(type(target).__name__, target.decode())
        else:
            return ast.Leaf(type(target).__name__, str(target))

    assert isinstance(target, python_ast.AST)
    for chname, chval in python_ast.iter_fields(target):
        if chname == "ctx":
            # ctx is omitted
            continue

        is_list = isinstance(chval, list)
        if chval is None:
            continue

        if is_list:
            if len(chval) == 0:
                base_type = python_ast.AST.__name__
                is_leaf = False
            else:
                base_type = base_ast_type(chval[0]).__name__
                is_leaf = is_builtin_type(chval[0])

            if is_leaf:
                parent_type = f"{base_type}__list"
            else:
                parent_type = base_type

            elements: List[ast.AST] = []
            for i, elem in enumerate(chval):
                c = to_ast(elem)
                if isinstance(c, ast.Leaf):
                    c = ast.Node(
                        parent_type, [ast.Field("token", base_type, c)])
                elements.append(c)
            fields.append(ast.Field(chname, parent_type, elements))
        else:
            base_type = base_ast_type(chval).__name__
            fields.append(ast.Field(chname, base_type,
                                    to_ast(chval)))
    return ast.Node(type_name, fields)
