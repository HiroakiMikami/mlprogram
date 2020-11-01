from mlprogram.languages import Kinds, Lexer, LexerWithLineNumber, Token


class MockLexer(Lexer):
    def tokenize(self, value):
        if value == "":
            return None
        return [Token(None, v, v) for v in value.split(" ")]

    def untokenize(self, value):
        if len(value) == 0:
            return None
        return " ".join([x.raw_value for x in value])


class TestLexerWithLineNumber(object):
    def test_tokenizer(self):
        lexer = LexerWithLineNumber(MockLexer())
        assert lexer.tokenize("foo bar") == [
            Token(Kinds.LineNumber(), 0, 0),
            Token(None, "foo", "foo"),
            Token(None, "bar", "bar")
        ]
        assert lexer.tokenize("foo\nbar") == [
            Token(Kinds.LineNumber(), 0, 0), Token(None, "foo", "foo"),
            Token(Kinds.LineNumber(), 1, 1), Token(None, "bar", "bar")
        ]
        assert lexer.tokenize("foo\n") is None

    def test_untokenize(self):
        lexer = LexerWithLineNumber(MockLexer())
        assert lexer.untokenize(lexer.tokenize("foo bar")) == "foo bar\n"
        assert lexer.untokenize(lexer.tokenize("foo\nbar")) == "foo\nbar\n"
