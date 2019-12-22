import unittest
import ast

from nl2prog.language.python import to_ast, parse, unparse
from nl2prog.language.ast import Node


class TestParse(unittest.TestCase):
    def test_parse_code(self):
        self.assertEqual(
            to_ast(ast.parse("y = x + 1").body[0]),
            parse("y = x + 1")
        )

    def test_invalid_code(self):
        self.assertEqual(None, parse("if True"))


class TestUnparse(unittest.TestCase):
    def test_unparse_ast(self):
        self.assertEqual(
            "\ny = (x + 1)\n", unparse(parse("y = x + 1"))
        )

    def test_invalid_ast(self):
        self.assertEqual(None, unparse(Node("USub", [])))


if __name__ == "__main__":
    unittest.main()
