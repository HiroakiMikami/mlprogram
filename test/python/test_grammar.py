import unittest
from src.python.grammar import *
from src.grammar import NodeType, Node, Rule

import ast
import numpy as np

Expr = NodeType("Expr", False)
expr = NodeType("expr", False)
expr_ = NodeType("expr", True)
Call = NodeType("Call", False)
Name = NodeType("Name", False)
Num = NodeType("Num", False)
str_ = NodeType("str", False)
Str = NodeType("Str", False)
int_ = NodeType("int", False)


class TestGrammar(unittest.TestCase):
    def test_builtin_type(self):
        self.assertFalse(is_builtin_type(ast.Return()))
        self.assertTrue(is_builtin_type(str))

    def test_builtin_node_type(self):
        self.assertFalse(is_builtin_node_type(NodeType("Expr", False)))
        self.assertFalse(is_builtin_node_type(NodeType("expr", False)))
        self.assertTrue(is_builtin_node_type(NodeType("str", False)))
        self.assertTrue(is_builtin_node_type(NodeType("identifier", False)))

    def test_to_sequence(self):
        s = to_sequence(ast.parse("f(g(0))").body[0])
        self.assertEqual(s, [
            Rule(ROOT, (Node("-", Expr), )),
            Rule(Expr, (Node("value", expr), )),
            Rule(expr, (Node("-", Call), )),
            Rule(Call, (Node("func", expr), Node("args", expr_))),
            Rule(expr, (Node("-", Name), )),
            Rule(Name, (Node("id", str_), )), "f", CLOSE_NODE,
            Rule(expr_, (Node("val0", expr), )),
            Rule(expr, (Node("-", Call), )),
            Rule(Call, (Node("func", expr), Node("args", expr_))),
            Rule(expr, (Node("-", Name), )),
            Rule(Name, (Node("id", str_), )), "g", CLOSE_NODE,
            Rule(expr_, (Node("val0", expr), )),
            Rule(expr, (Node("-", Num), )),
            Rule(Num, (Node("n", int_), )), "0", CLOSE_NODE
        ])
        s = to_sequence(ast.parse("'x y'").body[0])
        self.assertEqual(s, [
            Rule(ROOT, (Node("-", Expr), )),
            Rule(Expr, (Node("value", expr), )),
            Rule(expr, (Node("-", Str), )),
            Rule(Str, (Node("s", str_), )), "x", " ", "y", CLOSE_NODE
        ])

    def test_to_ast(self):
        import transpyle
        unparser = transpyle.python.unparser.NativePythonUnparser()
        """
        func(x, y)
        """
        sequence = [
            Rule(ROOT, (Node("-", Call), )),
            Rule(Call, (Node("func", expr), Node("args", expr_))),
            Rule(expr, (Node("-", Name), )),
            Rule(Name, (Node("id", str_), )), "func", CLOSE_NODE,
            Rule(expr_, (Node("val0", expr), Node("val1", expr))),
            Rule(expr, (Node("-", Name), )),
            Rule(Name, (Node("id", str_), )), "x", CLOSE_NODE,
            Rule(expr, (Node("-", Name), )),
            Rule(Name, (Node("id", str_), )), "y", CLOSE_NODE
        ]
        tree = to_ast(sequence)
        self.assertEqual(unparser.unparse(tree), "func(x, y)\n")
        """
        300
        """
        sequence = [
            Rule(ROOT, (Node("-", Num), )),
            Rule(Num, (Node("n", int_), )), "300", CLOSE_NODE
        ]
        tree2 = to_ast(sequence)
        self.assertEqual(unparser.unparse(tree2), "300\n")


if __name__ == "__main__":
    unittest.main()
