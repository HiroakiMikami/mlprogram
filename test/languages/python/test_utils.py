import unittest
import ast as python_ast

from mlprogram.languages.python.utils import is_builtin_type
from mlprogram.languages.python import IsSubtype


class TestIsBuiltinType(unittest.TestCase):
    def test_ast(self):
        self.assertFalse(is_builtin_type(python_ast.Expr()))

    def test_builtin(self):
        self.assertTrue(is_builtin_type(10))
        self.assertTrue(is_builtin_type(None))
        self.assertTrue(is_builtin_type(True))


class TestIsSubtype(unittest.TestCase):
    def test_ast(self):
        self.assertTrue(IsSubtype()("Expr", "AST"))
        self.assertFalse(IsSubtype()("Expr", "Name"))

    def test_builtin(self):
        self.assertTrue(IsSubtype()("int", "int"))
        self.assertFalse(IsSubtype()("str", "int"))

    def test_special_type(self):
        self.assertTrue(IsSubtype()("Expr__list", "Expr__list"))
        self.assertFalse(IsSubtype()("Expr__list", "Expr"))
        self.assertTrue(IsSubtype()("str__proxy", "str__proxy"))
        self.assertFalse(IsSubtype()("str__proxy", "str"))


if __name__ == "__main__":
    unittest.main()
