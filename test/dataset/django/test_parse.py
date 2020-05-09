import unittest
import ast

from mlprogram.language.python import to_ast
from mlprogram.dataset.django import parse


class TestParse(unittest.TestCase):
    def test_parse_code(self):
        self.assertEqual(
            to_ast(ast.parse("y = x + 1").body[0]),
            parse("y = x + 1")
        )

    def test_partial_code(self):
        self.assertEqual(
            to_ast(ast.parse("if True: pass\nelif False:\n  f(x)").body[0]),
            parse("elif False:\n f(x)")
        )
        self.assertEqual(
            to_ast(ast.parse("if True: pass\nelse:\n  f(x)").body[0]),
            parse("else:\n f(x)")
        )
        self.assertEqual(
            to_ast(ast.parse("try:\n  pass\nexcept: pass").body[0]),
            parse("try:")
        )

    def test_invalid_code(self):
        self.assertEqual(None, parse("if True"))


if __name__ == "__main__":
    unittest.main()
