import unittest
import ast as python_ast

from nl2prog.language.python.utils import is_builtin_type
from nl2prog.language.python import is_subtype


class TestIsBuiltinType(unittest.TestCase):
    def test_ast(self):
        self.assertFalse(is_builtin_type(python_ast.Expr()))

    def test_builtin(self):
        self.assertTrue(is_builtin_type(10))
        self.assertTrue(is_builtin_type(None))
        self.assertTrue(is_builtin_type(True))


class TestIsSubtype(unittest.TestCase):
    def test_ast(self):
        self.assertTrue(is_subtype("Expr", "AST"))
        self.assertFalse(is_subtype("Expr", "Name"))

    def test_builtin(self):
        self.assertTrue(is_subtype("int", "int"))
        self.assertFalse(is_subtype("str", "int"))


if __name__ == "__main__":
    unittest.main()
