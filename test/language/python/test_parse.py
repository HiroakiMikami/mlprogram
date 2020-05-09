import unittest
import ast

from nl2prog.language.python import to_ast, parse, unparse, ParseMode
from nl2prog.ast.ast import Node


class TestParse(unittest.TestCase):
    def test_parse_code(self):
        self.assertEqual(
            to_ast(ast.parse("y = x + 1").body[0]),
            parse("y = x + 1")
        )

    def test_invalid_code(self):
        self.assertEqual(None, parse("if True"))

    def test_mode(self):
        self.assertEqual(
            to_ast(ast.parse("xs = input().split()\nprint(','.join(xs))")),
            parse("xs = input().split()\nprint(','.join(xs))",
                  mode=ParseMode.Exec)
        )


class TestUnparse(unittest.TestCase):
    def test_unparse_ast(self):
        self.assertEqual(
            "\ny = (x + 1)\n", unparse(parse("y = x + 1"))
        )

    def test_invalid_ast(self):
        self.assertEqual(None, unparse(Node("USub", [])))

    def test_mode(self):
        self.assertEqual(
            "\nxs = input().split()\nprint(','.join(xs))\n",
            unparse(parse("xs = input().split()\nprint(','.join(xs))",
                          mode=ParseMode.Exec))
        )


if __name__ == "__main__":
    unittest.main()
