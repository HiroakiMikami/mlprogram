import unittest
import ast as python_ast

from nl2code.language.python._utils import is_builtin_type


class TestIsBuiltinType(unittest.TestCase):
    def test_ast(self):
        self.assertFalse(is_builtin_type(python_ast.Expr()))

    def test_builtin(self):
        self.assertTrue(is_builtin_type(10))
        self.assertTrue(is_builtin_type(None))
        self.assertTrue(is_builtin_type(True))


if __name__ == "__main__":
    unittest.main()
