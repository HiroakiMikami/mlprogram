import ast

from mlprogram.languages import Node
from mlprogram.languages.python import Parser
from mlprogram.languages.python.python_ast_to_ast import to_ast


class TestParser(object):
    def test_parse_code(self):
        assert to_ast(ast.parse("y = x + 1").body[0], lambda x: [x]) == \
            Parser(lambda x: [x]).parse("y = x + 1")

    def test_parse_invalid_code(self):
        assert Parser(lambda x: [x]).parse("if True") is None

    def test_parse_with_mode(self):
        assert to_ast(ast.parse("xs = input().split()\nprint(','.join(xs))"),
                      lambda x: [x]) == \
            Parser(lambda x: [x], mode="exec").parse(
                "xs = input().split()\nprint(','.join(xs))")

    def test_unparse_ast(self):
        parser = Parser(lambda x: [x])
        assert "\ny = (x + 1)\n" == parser.unparse(parser.parse("y = x + 1"))

    def test_unparse_invalid_ast(self):
        parser = Parser(lambda x: [x])
        assert parser.unparse(Node("USub", [])) is None

    def test_unparse_with_mode(self):
        parser = Parser(lambda x: [x], mode="exec")
        assert "\nxs = input().split()\nprint(','.join(xs))\n" == \
            parser.unparse(parser.parse(
                "xs = input().split()\nprint(','.join(xs))"))
