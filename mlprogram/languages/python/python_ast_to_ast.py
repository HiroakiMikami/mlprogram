import ast as python_ast
from typing import List, Type, Union, Callable

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


def to_ast(target: PythonAST,
           tokenize: Callable[[str], List[str]]) -> ast.AST:
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
    def to_value(target: PythonAST) -> Union[ast.AST, List[ast.AST]]:
        if is_builtin_type(target):
            # Builtin-type
            if isinstance(target, bytes):
                value = target.decode()
            else:
                value = str(target)
            tokens = tokenize(value)
            return list(map(
                lambda token: ast.Leaf(type(target).__name__, token),
                tokens))
        assert isinstance(target, python_ast.AST)
        type_name = type(target).__name__
        fields: List[ast.Field] = []

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
                    parent_type = f"{base_type}__proxy"
                else:
                    parent_type = base_type

                elements: List[ast.AST] = []
                for i, elem in enumerate(chval):
                    c = to_value(elem)
                    if is_leaf:
                        c = ast.Node(
                            parent_type, [ast.Field("token", base_type, c)])
                    assert isinstance(c, ast.AST)
                    elements.append(c)
                fields.append(ast.Field(chname, parent_type, elements))
            else:
                base_type = base_ast_type(chval).__name__
                fields.append(ast.Field(chname, base_type,
                                        to_value(chval)))
        return ast.Node(type_name, fields)

    value = to_value(target)
    assert isinstance(value, ast.AST)
    return value
