from mlprogram.languages import Token
from mlprogram.languages.c import Lexer


class TestLexer(object):
    def test_happy_path(self):
        lexer = Lexer()
        assert lexer.tokenize("int a = 0;") == [
            Token("INT", "int", "int"),
            Token("ID", "a", "a"),
            Token("EQUALS", "=", "="),
            Token("INT_CONST_OCT", "0", "0"),
            Token("SEMI", ";", ";")
        ]

    def test_invalid_program(self):
        lexer = Lexer()
        assert lexer.tokenize("int a = 0") == [
            Token("INT", "int", "int"),
            Token("ID", "a", "a"),
            Token("EQUALS", "=", "="),
            Token("INT_CONST_OCT", "0", "0")
        ]

    def test_multiple_lines(self):
        lexer = Lexer()
        assert lexer.tokenize("int a = 0\nint b;") == [
            Token("INT", "int", "int"),
            Token("ID", "a", "a"),
            Token("EQUALS", "=", "="),
            Token("INT_CONST_OCT", "0", "0"),
            Token("INT", "int", "int"),
            Token("ID", "b", "b"),
            Token("SEMI", ";", ";")
        ]

    def test_untokenize(self):
        lexer = Lexer()
        assert lexer.untokenize(lexer.tokenize("int a = 0;\nint b;")) == \
            "int a = 0 ; int b ;"
