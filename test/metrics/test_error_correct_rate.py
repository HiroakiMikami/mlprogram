import numpy as np

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
            acc(test_cases=[("foo", None)], actual=""))
        assert np.allclose(
            0.0,
            acc(test_cases=[("foo", None)], actual="foo"))

    def test_increasing_errors(self):
        acc = ErrorCorrectRate(MockAnalyzer(), MockInterpreter())
        assert np.allclose(
            0.0,
            acc(test_cases=[("foo", None)], actual="foobar"))

    def test_original_has_no_error(self):
        acc = ErrorCorrectRate(MockAnalyzer(), MockInterpreter())
        assert np.allclose(
            1.0,
            acc(test_cases=[("", None)], actual=""))
        assert np.allclose(
            0.0,
            acc(test_cases=[("", None)], actual="foobar"))
