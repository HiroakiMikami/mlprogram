import unittest
import ast as python_ast

import nl2code.language.ast as ast

from nl2code.language.python._python_ast_to_ast import base_ast_type
from nl2code.language.python import to_ast


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

    def test_variadic_args(self):
        node = python_ast.List()
        setattr(node, "elts", [python_ast.Num(), python_ast.Str()])
        self.assertEqual(
            ast.Node("List", [
                ast.Field("elts", "expr",
                          [ast.Node("Num", []), ast.Node("Str", [])])]),
            to_ast(node)
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
            ast.Node("Global", [ast.Field("names", "str__list", [
                ast.Node("str__list", [ast.Field("token", "str",
                                                 ast.Leaf("str", "v1"))]),
                ast.Node("str__list", [ast.Field("token", "str",
                                                 ast.Leaf("str", "v2"))])
            ])]),
            to_ast(node)
        )


if __name__ == "__main__":
    unittest.main()
