import unittest
import ast as python_ast

from nl2prog.language.action import NodeType
from nl2prog.language.python._utils import is_builtin_type
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
        self.assertTrue(is_subtype(
            NodeType("Expr", None),
            NodeType("AST", None)
        ))
        self.assertFalse(is_subtype(
            NodeType("Expr", None),
            NodeType("Name", None)
        ))

    def test_builtin(self):
        self.assertTrue(is_subtype(
            NodeType("int", None),
            NodeType("int", None)
        ))
        self.assertFalse(is_subtype(
            NodeType("str", None),
            NodeType("int", None)
        ))


if __name__ == "__main__":
    unittest.main()
