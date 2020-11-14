from mlprogram import Environment
from mlprogram.languages import Kinds, Lexer, Token
from mlprogram.languages.linediff import (
    Diff,
    Expander,
    Interpreter,
    IsSubtype,
    Parser,
    Remove,
    Replace,
    ToEpisode,
    get_samples,
)


class MockLexer(Lexer):
    def tokenize(self, code):
        return [Token(None, code, code)]


class TestIsSubType(object):
    def test_happy_path(self):
        assert IsSubtype()("Delta", "Delta")
        assert IsSubtype()("Insert", "Delta")
        assert IsSubtype()("Delta", "Delta")
        assert IsSubtype()("str", "value")
        assert not IsSubtype()(Kinds.LineNumber(), "value")


class TestGetSamples(object):
    def test_happy_path(self):
        samples = get_samples(Parser(MockLexer()))
        assert len(samples.rules) == 5
        assert len(samples.tokens) == 0


class TestToEpisode(object):
    def test_happy_path(self):
        to_episode = ToEpisode(Interpreter(), Expander())
        episode = to_episode(Environment(
            inputs={"text_query": "xxx\nyyy"},
            supervisions={"ground_truth": Diff([Replace(0, "zzz"), Remove(1)])}
        ))
        assert len(episode) == 2
        assert episode[0].inputs["code"] == "xxx\nyyy"
        assert episode[1].inputs["code"] == "zzz\nyyy"
