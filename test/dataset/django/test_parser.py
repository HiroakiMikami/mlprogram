import ast

from mlprogram.languages.python.python_ast_to_ast import to_ast
from mlprogram.datasets.django import Parser


class TestParse(object):
    def test_parse_code(self):
        assert to_ast(ast.parse("y = x + 1").body[0], lambda x: [x]) == \
            Parser(lambda x: [x]).parse("y = x + 1")

    def test_partial_code(self):
        assert to_ast(ast.parse("if True: pass\nelif False:\n  f(x)").body[0],
                      lambda x: [x]) == \
            Parser(lambda x: [x]).parse("elif False:\n f(x)")
        assert to_ast(ast.parse("if True: pass\nelse:\n  f(x)").body[0],
                      lambda x: [x]) == \
            Parser(lambda x: [x]).parse("else:\n f(x)")
        assert to_ast(ast.parse("try:\n  pass\nexcept: pass").body[0],
                      lambda x: [x]) == \
            Parser(lambda x: [x]).parse("try:")

    def test_invalid_code(self):
        assert Parser(lambda x: [x]).parse("if True") is None
