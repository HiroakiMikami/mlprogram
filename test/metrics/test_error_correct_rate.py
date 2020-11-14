import numpy as np

from mlprogram import Environment
from mlprogram.languages import Analyzer, Interpreter
from mlprogram.metrics import ErrorCorrectRate


class MockAnalyzer(Analyzer):
    def __call__(self, value):
        return list(value)


class MockInterpreter(Interpreter):
    def eval(self, value, inputs):
        return [value]


class TestErrorCorrectRate(object):
    def test_simple_case(self):
        acc = ErrorCorrectRate(MockAnalyzer(), MockInterpreter())
        assert np.allclose(
            1.0,
            acc(Environment(inputs={"text_query": "foo"}), ""))
        assert np.allclose(
            0.0,
            acc(Environment(inputs={"text_query": "foo"}), "foo"))

    def test_increasing_errors(self):
        acc = ErrorCorrectRate(MockAnalyzer(), MockInterpreter())
        assert np.allclose(
            0.0,
            acc(Environment(inputs={"text_query": "foo"}), "foobar"))
