import ast as python_ast
from typing import List

import nl2code.language.ast as ast
from nl2code.language.python import PythonAST
from ._utils import is_builtin_type


def base_ast_type(node: PythonAST):
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
        if type(target) == bytes:
            return ast.Leaf(type(target).__name__, target.decode())
        else:
            return ast.Leaf(type(target).__name__, str(target))

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
            else:
                base_type = base_ast_type(chval[0]).__name__

            # TODO use variadic fields
            elements: List[ast.Field] = []
            for i, elem in enumerate(chval):
                elements.append(ast.Field("val__{}".format(i), base_type,
                                          to_ast(elem)))
            fields.append(ast.Field(chname, "{}__list".format(base_type),
                                    ast.Node("{}__list".format(base_type),
                                             elements)))
        else:
            base_type = base_ast_type(chval).__name__
            fields.append(ast.Field(chname, base_type,
                                    to_ast(chval)))
    return ast.Node(type_name, fields)
