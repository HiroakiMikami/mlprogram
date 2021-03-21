from mlprogram.languages import Token
from mlprogram.languages.c import Lexer


class TestLexer(object):
    def test_happy_path(self):
        lexer = Lexer()
        assert lexer.tokenize_with_offset("int a = 0;") == [
            (0, Token("INT", "int", "int")),
            (4, Token("ID", "a", "a")),
            (6, Token("EQUALS", "=", "=")),
            (8, Token("INT_CONST_OCT", "0", "0")),
            (9, Token("SEMI", ";", ";"))
        ]

    def test_invalid_program(self):
        lexer = Lexer()
        assert lexer.tokenize_with_offset("int a = 0") == [
            (0, Token("INT", "int", "int")),
            (4, Token("ID", "a", "a")),
            (6, Token("EQUALS", "=", "=")),
            (8, Token("INT_CONST_OCT", "0", "0")),
        ]

    def test_multiple_lines(self):
        lexer = Lexer()
        assert lexer.tokenize_with_offset("int a = 0\nint b;") == [
            (0, Token("INT", "int", "int")),
            (4, Token("ID", "a", "a")),
            (6, Token("EQUALS", "=", "=")),
            (8, Token("INT_CONST_OCT", "0", "0")),
            (10, Token("INT", "int", "int")),
            (14, Token("ID", "b", "b")),
            (15, Token("SEMI", ";", ";"))
        ]

    def test_untokenize(self):
        lexer = Lexer()
        assert lexer.untokenize(lexer.tokenize("int a = 0;\nint b;")) == \
            "int a = 0 ; int b ;"

    def test_escape_sequence(self):
        lexer = Lexer()
        assert lexer.untokenize(lexer.tokenize("int a = 0;\r\nint b;")) == \
            "int a = 0 ; int b ;"
