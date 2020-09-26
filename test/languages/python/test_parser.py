import unittest
import ast

from mlprogram.languages.python.python_ast_to_ast import to_ast
from mlprogram.languages.python import Parser
from mlprogram.languages import Node


class TestParser(unittest.TestCase):
    def test_parse_code(self):
        self.assertEqual(
            to_ast(ast.parse("y = x + 1").body[0], lambda x: [x]),
            Parser(lambda x: [x]).parse("y = x + 1")
        )

    def test_parse_invalid_code(self):
        self.assertEqual(None, Parser(lambda x: [x]).parse("if True"))

    def test_parse_with_mode(self):
        self.assertEqual(
            to_ast(ast.parse("xs = input().split()\nprint(','.join(xs))"),
                   lambda x: [x]),
            Parser(lambda x: [x], mode="exec").parse(
                "xs = input().split()\nprint(','.join(xs))")
        )

    def test_unparse_ast(self):
        parser = Parser(lambda x: [x])
        self.assertEqual(
            "\ny = (x + 1)\n", parser.unparse(parser.parse("y = x + 1"))
        )

    def test_unparse_invalid_ast(self):
        parser = Parser(lambda x: [x])
        self.assertEqual(None, parser.unparse(Node("USub", [])))

    def test_unparse_with_mode(self):
        parser = Parser(lambda x: [x], mode="exec")
        self.assertEqual(
            "\nxs = input().split()\nprint(','.join(xs))\n",
            parser.unparse(parser.parse(
                "xs = input().split()\nprint(','.join(xs))"))
        )


if __name__ == "__main__":
    unittest.main()
