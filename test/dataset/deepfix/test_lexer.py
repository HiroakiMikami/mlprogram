from mlprogram.datasets.deepfix import Lexer
from mlprogram.languages import Token


class TestLexer(object):
    def test_id_placeholder(self):
        lexer = Lexer()
        assert lexer.tokenize_with_offset("int a;a;b;") == [
            (0, Token("INT", "int", "int")),
            (4, Token("ID", "___id@0___", "a")),
            (5, Token("SEMI", ";", ";")),
            (6, Token("ID", "___id@0___", "a")),
            (7, Token("SEMI", ";", ";")),
            (8, Token("ID", "___id@1___", "b")),
            (9, Token("SEMI", ";", ";"))
        ]

    def test_int_placeholder(self):
        lexer = Lexer()
        assert lexer.tokenize_with_offset("int a = 1;") == [
            (0, Token("INT", "int", "int")),
            (4, Token("ID", "___id@0___", "a")),
            (6, Token("EQUALS", "=", "=")),
            (8, Token("INT_CONST_DEC", "___int@0___", "1")),
            (9, Token("SEMI", ";", ";"))
        ]

    def test_float_placeholder(self):
        lexer = Lexer()
        assert lexer.tokenize_with_offset("float a = 1.0;") == [
            (0, Token("FLOAT", "float", "float")),
            (6, Token("ID", "___id@0___", "a")),
            (8, Token("EQUALS", "=", "=")),
            (10, Token("FLOAT_CONST", "___float@0___", "1.0")),
            (13, Token("SEMI", ";", ";"))
        ]

    def test_str_placeholder(self):
        lexer = Lexer()
        assert lexer.tokenize_with_offset("\"foo\"") == [
            (0, Token("STRING_LITERAL", "___string@0___", "\"foo\"")),
        ]

    def test_char_placeholder(self):
        lexer = Lexer()
        assert lexer.tokenize_with_offset("'a'") == [
            (0, Token("CHAR_CONST", "___char@0___", "'a'")),
        ]
