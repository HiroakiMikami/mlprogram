from mlprogram.languages import BatchedState
from mlprogram.languages.linediff import Diff, Insert, Interpreter, Remove, Replace


class TestInterpreter(object):
    def test_eval(self):
        interpreter = Interpreter()
        assert interpreter.eval(Insert(0, "foo"), ["bar\nhoge"]) == \
            ["foo\nbar\nhoge"]
        assert interpreter.eval(Remove(0), ["bar\nhoge"]) == \
            ["hoge"]
        assert interpreter.eval(Replace(0, "foo"), ["bar\nhoge"]) == \
            ["foo\nhoge"]

    def test_eval_diff(self):
        interpreter = Interpreter()
        assert interpreter.eval(
            Diff([Insert(0, "foo"), Replace(1, "test")]), ["bar\nhoge"]
        ) == ["foo\ntest\nhoge"]

    def test_execute(self):
        ref0 = Insert(0, "foo")
        ref1 = Replace(1, "test")
        state = BatchedState({}, {}, [], ["bar\nhoge"])
        interpreter = Interpreter()

        state = interpreter.execute(ref0, state)
        assert state.history == [ref0]
        assert set(state.environment.keys()) == set([ref0])
        assert state.type_environment[ref0] == "Insert"
        assert state.environment[ref0][0] == "foo\nbar\nhoge"
        assert state.context[0] == "foo\nbar\nhoge"

        state = interpreter.execute(ref1, state)
        assert state.history == [ref0, ref1]
        assert set(state.environment.keys()) == set([ref1])
        assert state.type_environment[ref1] == "Replace"
        assert state.environment[ref1][0] == "foo\ntest\nhoge"
        assert state.context[0] == "foo\ntest\nhoge"
