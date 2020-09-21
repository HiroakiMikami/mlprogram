import unittest
from mlprogram.languages.c import Token, Tokenizer


class TestTokenizer(unittest.TestCase):
    def test_happy_path(self):
        tokenizer = Tokenizer()
        self.assertEqual(
            [Token("int", 0, "INT"), Token("a", 4, "ID"),
             Token("=", 6, "EQUALS"), Token("0", 8, "INT_CONST_OCT"),
             Token(";", 9, "SEMI")],
            tokenizer("int a = 0;"))

    def test_invalid_program(self):
        tokenizer = Tokenizer()
        self.assertEqual(
            [Token("int", 0, "INT"), Token("a", 4, "ID"),
             Token("=", 6, "EQUALS"), Token("0", 8, "INT_CONST_OCT")],
            tokenizer("int a = 0"))

    def test_multiple_lines(self):
        tokenizer = Tokenizer()
        self.assertEqual(
            [Token("int", 0, "INT"), Token("a", 4, "ID"),
             Token("=", 6, "EQUALS"), Token("0", 8, "INT_CONST_OCT"),
             Token("int", 10, "INT"), Token("b", 14, "ID"),
             Token(";", 15, "SEMI")],
            tokenizer("int a = 0\nint b;"))


if __name__ == "__main__":
    unittest.main()
