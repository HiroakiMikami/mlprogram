import ast as python_ast

import mlprogram.languages as ast
from mlprogram.languages.python.python_ast_to_ast import base_ast_type, to_ast


class TestBaseASTType(object):
    def test_base_type(self):
        assert python_ast.stmt == base_ast_type(python_ast.stmt())

    def test_sub_type(self):
        assert python_ast.stmt == base_ast_type(python_ast.Expr())

    def test_builtin_type(self):
        assert int == base_ast_type(10)


class TestToAST(object):
    def test_node(self):
        node = python_ast.Expr()
        setattr(node, "value", python_ast.Name())
        assert ast.Node("Expr", [ast.Field("value", "expr",
                                           ast.Node("Name", []))]) == \
            to_ast(node, lambda x: [x])

    def test_node_with_ctx_field(self):
        node = python_ast.List()
        setattr(node, "ctx", 10)
        assert ast.Node("List", []) == to_ast(node, lambda x: [x])

    def test_tokenize_builtin_type(self):
        def split_value(x):
            return list(x)

        node = python_ast.List()
        ten = python_ast.Constant()
        setattr(ten, "value", 10)
        setattr(node, "elts", [ten])
        assert ast.Node(
            "List",
            [ast.Field("elts", "expr",
                       [ast.Node("Constant",
                                 [ast.Field("value", "int",
                                            [ast.Leaf("int", "1"),
                                             ast.Leaf("int", "0")])])
                        ])]) == to_ast(node, split_value)

    def test_variadic_args(self):
        node = python_ast.List()
        setattr(node, "elts", [python_ast.Num(), python_ast.Str()])
        assert ast.Node(
            "List", [
                ast.Field("elts", "expr",
                          [ast.Node("Constant", []),
                           ast.Node("Constant", [])])]
        ) == to_ast(node, lambda x: [x])

    def test_optional_arg(self):
        node = python_ast.Yield()
        setattr(node, "value", None)
        assert ast.Node("Yield", []) == to_ast(node, lambda x: [x])

    def test_empty_list(self):
        node = python_ast.List()
        setattr(node, "elts", [])
        assert ast.Node("List", [ast.Field("elts", "AST", [])]) == \
            to_ast(node, lambda x: [x])

    def test_token_list(self):
        node = python_ast.Global()
        setattr(node, "names", ["v1", "v2"])
        assert ast.Node("Global", [ast.Field("names", "str__proxy", [
            ast.Node("str__proxy", [ast.Field("token", "str",
                                              [ast.Leaf("str", "v1")])]),
            ast.Node("str__proxy", [ast.Field("token", "str",
                                              [ast.Leaf("str", "v2")])])
        ])]) == \
            to_ast(node, lambda x: [x])
