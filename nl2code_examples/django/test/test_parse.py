import unittest
import ast

from nl2code_examples.django import parse, unparse


class TestParse(unittest.TestCase):
    def test_parse_code(self):
        self.assertEqual(
            ast.dump(ast.parse("y = x + 1").body[0]),
            ast.dump(parse("y = x + 1"))
        )

    def test_partial_code(self):
        self.assertEqual(
            ast.dump(ast.parse("if True: pass\nelif False:\n  f(x)").body[0]),
            ast.dump(parse("elif False:\n f(x)"))
        )
        self.assertEqual(
            ast.dump(ast.parse("if True: pass\nelse:\n  f(x)").body[0]),
            ast.dump(parse("else:\n f(x)"))
        )
        self.assertEqual(
            ast.dump(ast.parse("try:\n  pass\nexcept: pass").body[0]),
            ast.dump(parse("try:"))
        )


class TestUnparse(unittest.TestCase):
    def test_unparse_ast(self):
        self.assertEqual(
            "\ny = (x + 1)\n", unparse(parse("y = x + 1"))
        )


if __name__ == "__main__":
    unittest.main()
