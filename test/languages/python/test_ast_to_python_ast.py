import unittest
import ast as python_ast

import mlprogram.asts as ast

from mlprogram.languages.python.ast_to_python_ast import to_builtin_type
from mlprogram.languages.python import to_ast, to_python_ast


class TestToBuiltinType(unittest.TestCase):
    def test_int(self):
        self.assertEqual(10, to_builtin_type("10", "int"))

    def test_bool(self):
        self.assertEqual(True, to_builtin_type("True", "bool"))
        self.assertEqual(False, to_builtin_type("False", "bool"))

    def test_bytes(self):
        self.assertEqual(bytes([10, 10]),
                         to_builtin_type(bytes([10, 10]).decode(), "bytes"))

    def test_None(self):
        self.assertEqual(None, to_builtin_type("None", "NoneType"))


class TestToPythonAST(unittest.TestCase):
    def test_node(self):
        node = python_ast.Expr()
        name = python_ast.Name()
        setattr(name, "id", None)
        setattr(name, "ctx", None)
        setattr(node, "value", name)
        node2 = to_python_ast(to_ast(node))
        self.assertEqual(python_ast.dump(node), python_ast.dump(node2))

    def test_builtin_type(self):
        self.assertEqual(10, to_python_ast(ast.Leaf("int", "10")))
        node = python_ast.List()
        ten = python_ast.Constant()
        setattr(ten, "value", 10)
        setattr(ten, "kind", None)
        setattr(node, "elts", [ten])
        setattr(node, "ctx", None)
        self.assertEqual(python_ast.dump(node),
                         python_ast.dump(to_python_ast(to_ast(node,
                                                              tokenize=list))))
        self.assertEqual(True, to_python_ast(ast.Leaf("bool", "True")))

    def test_variadic_args(self):
        node = python_ast.List()
        n = python_ast.Constant()
        s = python_ast.Constant()
        setattr(n, "n", 10)
        setattr(n, "kind", None)
        setattr(s, "s", "foo")
        setattr(s, "kind", None)
        setattr(node, "elts", [n, s])
        setattr(node, "ctx", None)
        self.assertEqual(
            python_ast.dump(node),
            python_ast.dump(to_python_ast(to_ast(node))))

    def test_optional_arg(self):
        node = python_ast.Yield()
        setattr(node, "value", None)
        self.assertEqual(python_ast.dump(node),
                         python_ast.dump(to_python_ast(to_ast(node))))

    def test_empty_list(self):
        node = python_ast.List()
        setattr(node, "ctx", None)
        setattr(node, "elts", [])
        self.assertEqual(
            python_ast.dump(node),
            python_ast.dump(to_python_ast(to_ast(node))))

    def test_token_list(self):
        node = python_ast.Global()
        setattr(node, "names", ["v1", "v2"])
        self.assertEqual(
            python_ast.dump(node),
            python_ast.dump(to_python_ast(to_ast(node))))


if __name__ == "__main__":
    unittest.main()
