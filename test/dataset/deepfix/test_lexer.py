from mlprogram.datasets.deepfix import Lexer
from mlprogram.languages import Token


class TestLexer(object):
    def test_id_placeholder(self):
        lexer = Lexer()
        assert lexer.tokenize("int a;a;b;") == [
            Token("INT", "int", "int"),
            Token("ID", "___id@0___", "a"),
            Token("SEMI", ";", ";"),
            Token("ID", "___id@0___", "a"),
            Token("SEMI", ";", ";"),
            Token("ID", "___id@1___", "b"),
            Token("SEMI", ";", ";")
        ]

    def test_int_placeholder(self):
        lexer = Lexer()
        assert lexer.tokenize("int a = 1;") == [
            Token("INT", "int", "int"),
            Token("ID", "___id@0___", "a"),
            Token("EQUALS", "=", "="),
            Token("INT_CONST_DEC", "___int@0___", "1"),
            Token("SEMI", ";", ";")
        ]

    def test_float_placeholder(self):
        lexer = Lexer()
        assert lexer.tokenize("float a = 1.0;") == [
            Token("FLOAT", "float", "float"),
            Token("ID", "___id@0___", "a"),
            Token("EQUALS", "=", "="),
            Token("FLOAT_CONST", "___float@0___", "1.0"),
            Token("SEMI", ";", ";")
        ]

    def test_str_placeholder(self):
        lexer = Lexer()
        assert lexer.tokenize("\"foo\"") == [
            Token("STRING_LITERAL", "___string@0___", "\"foo\""),
        ]

    def test_char_placeholder(self):
        lexer = Lexer()
        assert lexer.tokenize("'a'") == [
            Token("CHAR_CONST", "___char@0___", "'a'"),
        ]
