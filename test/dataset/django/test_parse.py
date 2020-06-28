import unittest
import ast

from mlprogram.languages.python import to_ast
from mlprogram.datasets.django import Parse


class TestParse(unittest.TestCase):
    def test_parse_code(self):
        self.assertEqual(
            to_ast(ast.parse("y = x + 1").body[0]),
            Parse()("y = x + 1")
        )

    def test_partial_code(self):
        self.assertEqual(
            to_ast(ast.parse("if True: pass\nelif False:\n  f(x)").body[0]),
            Parse()("elif False:\n f(x)")
        )
        self.assertEqual(
            to_ast(ast.parse("if True: pass\nelse:\n  f(x)").body[0]),
            Parse()("else:\n f(x)")
        )
        self.assertEqual(
            to_ast(ast.parse("try:\n  pass\nexcept: pass").body[0]),
            Parse()("try:")
        )

    def test_invalid_code(self):
        self.assertEqual(None, Parse()("if True"))


if __name__ == "__main__":
    unittest.main()
