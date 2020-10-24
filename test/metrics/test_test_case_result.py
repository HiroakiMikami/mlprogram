import numpy as np
from mlprogram import Environment
from mlprogram.metrics import TestCaseResult
from mlprogram.interpreters import Interpreter


class MockInterpreter(Interpreter):
    def eval(self, code: str, inputs):
        return [int(code) + input for input in inputs]

    def eval_references(self, code, inputs):
        return [{stmt.reference: int(stmt.code) + input
                for stmt in code.statements} for input in inputs]


class TestTestCaseResult(object):
    def test_simple_case(self):
        acc = TestCaseResult(MockInterpreter())
        assert np.allclose(1.0,
                           acc(Environment(inputs={"test_cases": [(1, 0)]}),
                               "-1"))
        assert np.allclose(0.0,
                           acc(Environment(inputs={"test_cases": [(1, 0)]}),
                               "2"))
        assert np.allclose(1.0,
                           acc(Environment(inputs={"test_cases": [(1, 1)]}),
                               "0"))

    def test_custom_metric(self):
        def metric(input, actual):
            return input.supervisions["ground_truth"] + actual

        acc = TestCaseResult(MockInterpreter(), metric=metric)
        assert np.allclose(0.0,
                           acc(Environment(inputs={"test_cases": [(0, 0)]}),
                               "0"))
        assert np.allclose(1.0,
                           acc(Environment(inputs={"test_cases": [(0, 0)]}),
                               "1"))
        assert np.allclose(2.0,
                           acc(Environment(inputs={"test_cases": [(0, 1)]}),
                               "1"))
