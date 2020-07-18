import unittest
import ast as python_ast

import mlprogram.asts as ast

from mlprogram.languages.python.python_ast_to_ast import base_ast_type
from mlprogram.languages.python import to_ast


class TestBaseAstType(unittest.TestCase):
    def test_base_type(self):
        self.assertEqual(python_ast.stmt, base_ast_type(python_ast.stmt()))

    def test_sub_type(self):
        self.assertEqual(python_ast.stmt, base_ast_type(python_ast.Expr()))

    def test_builtin_type(self):
        self.assertEqual(int, base_ast_type(10))


class TestToAST(unittest.TestCase):
    def test_node(self):
        node = python_ast.Expr()
        setattr(node, "value", python_ast.Name())
        self.assertEqual(
            ast.Node("Expr", [ast.Field("value", "expr",
                                        ast.Node("Name", []))]),
            to_ast(node))

    def test_node_with_ctx_field(self):
        node = python_ast.List()
        setattr(node, "ctx", 10)
        self.assertEqual(ast.Node("List", []), to_ast(node))

    def test_builtin_type(self):
        self.assertEqual(ast.Leaf("int", "10"), to_ast(10))
        self.assertEqual(ast.Leaf("bool", "True"), to_ast(True))

    def test_tokenize_builtin_type(self):
        def tokenize(x: str):
            return list(x)

        node = python_ast.List()
        ten = python_ast.Constant()
        setattr(ten, "value", 10)
        setattr(node, "elts", [ten])
        self.assertEqual(
            ast.Node("List",
                     [ast.Field("elts", "expr",
                                [ast.Node("Constant",
                                          [ast.Field("value", "int",
                                                     [ast.Leaf("int", "1"),
                                                      ast.Leaf("int", "0")])])
                                 ])]),
            to_ast(node, tokenize))

    def test_variadic_args(self):
        node = python_ast.List()
        setattr(node, "elts", [python_ast.Num(), python_ast.Str()])
        self.assertEqual(
            ast.Node("List", [
                ast.Field("elts", "expr",
                          [ast.Node("Constant", []),
                           ast.Node("Constant", [])])]),
            to_ast(node)
        )

    def test_expand_variadic_args(self):
        node = python_ast.List()
        setattr(node, "elts", [python_ast.Num(), python_ast.Str()])
        self.assertEqual(
            ast.Node("List", [
                ast.Field(
                    "elts", "expr__list",
                    ast.Node("expr__list", [
                        ast.Field("0", "expr", ast.Node("Constant", [])),
                        ast.Field("1", "expr", ast.Node("Constant", [])),
                    ])
                )
            ]),
            to_ast(node, retain_variadic_fields=False)
        )

    def test_optional_arg(self):
        node = python_ast.Yield()
        setattr(node, "value", None)
        self.assertEqual(ast.Node("Yield", []), to_ast(node))

    def test_empty_list(self):
        node = python_ast.List()
        setattr(node, "elts", [])
        self.assertEqual(
            ast.Node("List", [ast.Field("elts", "AST", [])]),
            to_ast(node))

    def test_token_list(self):
        node = python_ast.Global()
        setattr(node, "names", ["v1", "v2"])
        self.assertEqual(
            ast.Node("Global", [ast.Field("names", "str__proxy", [
                ast.Node("str__proxy", [ast.Field("token", "str",
                                                  ast.Leaf("str", "v1"))]),
                ast.Node("str__proxy", [ast.Field("token", "str",
                                                  ast.Leaf("str", "v2"))])
            ])]),
            to_ast(node)
        )

    def test_expand_token_list(self):
        node = python_ast.Global()
        setattr(node, "names", ["v1", "v2"])
        self.assertEqual(
            ast.Node("Global", [
                ast.Field("names", "str__proxy__list",
                          ast.Node("str__proxy__list", [
                              ast.Field("0", "str__proxy",
                                        ast.Node("str__proxy", [
                                            ast.Field("token", "str",
                                                      ast.Leaf("str", "v1"))
                                        ])),
                              ast.Field("1", "str__proxy",
                                        ast.Node("str__proxy", [
                                            ast.Field("token", "str",
                                                      ast.Leaf("str", "v2"))
                                        ]))
                          ]))]),
            to_ast(node, retain_variadic_fields=False)
        )


if __name__ == "__main__":
    unittest.main()
