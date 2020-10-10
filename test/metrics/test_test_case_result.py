import numpy as np
from mlprogram.metrics import TestCaseResult
from mlprogram.interpreters import Reference
from mlprogram.interpreters import Statement
from mlprogram.interpreters import SequentialProgram
from mlprogram.interpreters import Interpreter


class MockInterpreter(Interpreter):
    def eval(self, code: str, input) -> int:
        return int(code) + input

    def eval_references(self, code, input):
        return {stmt.reference: int(stmt.code) + input
                for stmt in code.statements}


class TestTestCaseResult(object):
    def test_simple_case(self):
        acc = TestCaseResult(MockInterpreter())
        assert np.allclose(1.0, acc({"input": (1, 0)}, "-1"))
        assert np.allclose(0.0, acc({"input": (1, 0)}, "2"))
        assert np.allclose(1.0, acc({"input": (1, 1)}, "0"))

    def test_reference(self):
        acc = TestCaseResult(MockInterpreter(), reference=True)
        assert np.allclose(
            1.0,
            acc({"input": (0, 0)},
                SequentialProgram([Statement(Reference("0"), "0")]))
        )

    def test_custom_metric(self):
        def metric(input, actual):
            return input["ground_truth"] + actual

        acc = TestCaseResult(MockInterpreter(), metric=metric)
        assert np.allclose(0.0, acc({"input": (0, 0)}, "0"))
        assert np.allclose(1.0, acc({"input": (0, 0)}, "1"))
        assert np.allclose(2.0, acc({"input": (0, 1)}, "1"))
