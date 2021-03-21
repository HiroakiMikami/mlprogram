from mlprogram.datasets.deepfix import Lexer
from mlprogram.languages import Token


class TestLexer(object):
    def test_id_placeholder(self):
        lexer = Lexer()
        assert lexer.tokenize_with_offset("int a;a;b;") == [
            (0, Token("name", "___name@0___", "int")),
            (4, Token("name", "___name@1___", "a")),
            (5, Token("op", ";", ";")),
            (6, Token("name", "___name@1___", "a")),
            (7, Token("op", ";", ";")),
            (8, Token("name", "___name@2___", "b")),
            (9, Token("op", ";", ";"))
        ]

    def test_int_placeholder(self):
        lexer = Lexer()
        assert lexer.tokenize_with_offset("int a = 1;") == [
            (0, Token("name", "___name@0___", "int")),
            (4, Token("name", "___name@1___", "a")),
            (6, Token("op", "=", "=")),
            (8, Token("number", "___number@0___", "1")),
            (9, Token("op", ";", ";"))
        ]

    def test_float_placeholder(self):
        lexer = Lexer()
        assert lexer.tokenize_with_offset("float a = 1.0;") == [
            (0, Token("name", "___name@0___", "float")),
            (6, Token("name", "___name@1___", "a")),
            (8, Token("op", "=", "=")),
            (10, Token("number", "___number@0___", "1.0")),
            (13, Token("op", ";", ";"))
        ]

    def test_str_placeholder(self):
        lexer = Lexer()
        assert lexer.tokenize_with_offset("\"foo\"") == [
            (0, Token("string", "___string@0___", "\"foo\"")),
        ]

    def test_char_placeholder(self):
        lexer = Lexer()
        assert lexer.tokenize_with_offset("'a'") == [
            (0, Token("char", "___char@0___", "'a'")),
        ]

    def test_untokenize(self):
        lexer = Lexer()
        assert lexer.untokenize(lexer.tokenize("int x = 0;")) == \
            "int x = 0 ;"
