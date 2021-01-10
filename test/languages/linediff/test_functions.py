from mlprogram.builtins import Environment
from mlprogram.languages import BatchedState, Kinds, Lexer, Token
from mlprogram.languages.linediff import (
    AddTestCases,
    Diff,
    Expander,
    Interpreter,
    IsSubtype,
    Parser,
    Remove,
    Replace,
    ToEpisode,
    UpdateInput,
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
            {"test_cases": [("xxx\nyyy", None)],
             "ground_truth": Diff([Replace(0, "zzz"), Remove(1)])},
            set(["ground_truth"])
        ))
        assert len(episode) == 2
        assert episode[0]["interpreter_state"].context == ["xxx\nyyy"]
        assert episode[1]["interpreter_state"].context == ["zzz\nyyy"]


class TestAddTestCases(object):
    def test_happy_path(self):
        f = AddTestCases()
        entry = f(Environment({"code": "xxx\nyyy"}))
        assert entry["test_cases"] == [("xxx\nyyy", None)]


class TestUpdateInput(object):
    def test_happy_path(self):
        f = UpdateInput()
        entry = f(Environment({
            "interpreter_state": BatchedState({}, {}, [], ["xxx\nyyy"])
        }))
        assert entry["code"] == "xxx\nyyy"
        state = BatchedState({}, {Diff([]): ["foo"]}, [Diff([])], ["foo"])
        entry = f(Environment({"interpreter_state": state}))
        assert entry["code"] == "foo"
