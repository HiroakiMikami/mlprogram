from mlprogram.languages import Kinds, Lexer, LexerWithLineNumber, Token


class MockLexer(Lexer):
    def tokenize_with_offset(self, value):
        if value == "\n":
            return None
        retval = []
        offset = 0
        while True:
            next_sp = value.find(" ", offset)
            next_nl = value.find("\n", offset)
            if next_sp == -1 and next_nl == -1:
                retval.append((offset, Token(None, value[offset:], value[offset:])))
                break
            next_sp = next_sp if next_sp >= 0 else len(value) + 1
            next_nl = next_nl if next_nl >= 0 else len(value) + 1
            next = min(next_sp, next_nl)
            v = value[offset:next]
            retval.append((offset, Token(None, v, v)))
            offset = next + 1
        return retval

    def untokenize(self, value):
        if len(value) == 0:
            return None
        return " ".join([x.raw_value for x in value])


class TestLexerWithLineNumber(object):
    def test_tokenizer(self):
        lexer = LexerWithLineNumber(MockLexer())
        assert lexer.tokenize_with_offset("foo bar") == [
            (0, Token(Kinds.LineNumber(), 0, 0)),
            (0, Token(None, "foo", "foo")),
            (4, Token(None, "bar", "bar")),
            (8, Token(Kinds.LineNumber(), 1, 1)),
            (8, Token(None, "", "")),
        ]
        assert lexer.tokenize_with_offset("foo\nbar") == [
            (0, Token(Kinds.LineNumber(), 0, 0)),
            (0, Token(None, "foo", "foo")),
            (4, Token(Kinds.LineNumber(), 1, 1)),
            (4, Token(None, "bar", "bar")),
            (8, Token(Kinds.LineNumber(), 2, 2)),
            (8, Token(None, "", "")),
        ]
        assert lexer.tokenize_with_offset("foo\nbar\n") == [
            (0, Token(Kinds.LineNumber(), 0, 0)),
            (0, Token(None, "foo", "foo")),
            (4, Token(Kinds.LineNumber(), 1, 1)),
            (4, Token(None, "bar", "bar")),
            (8, Token(Kinds.LineNumber(), 2, 2)),
            (8, Token(None, "", "")),
        ]
        assert lexer.tokenize("") is None

    def test_untokenize(self):
        lexer = LexerWithLineNumber(MockLexer())
        assert lexer.untokenize(lexer.tokenize("foo bar")) == "foo bar\n"
        assert lexer.untokenize(lexer.tokenize("foo\nbar")) == "foo\nbar\n"
