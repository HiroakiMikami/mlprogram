from mlprogram.languages import Lexer, Token
from mlprogram.languages.linediff import IsSubtype, get_samples, Parser


class MockLexer(Lexer):
    def tokenize(self, code):
        return [Token(None, code, code)]


class TestIsSubType(object):
    def test_happy_path(self):
        assert IsSubtype()("Delta", "Delta")
        assert IsSubtype()("Insert", "Delta")


class TestGetSamples(object):
    def test_happy_path(self):
        samples = get_samples(Parser(MockLexer()))
        assert len(samples.rules) == 5
        assert len(samples.tokens) == 0
