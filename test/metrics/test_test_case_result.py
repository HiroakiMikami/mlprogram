import unittest
from mlprogram.metrics import TestCaseResult
from mlprogram.interpreters import Reference
from mlprogram.interpreters import Statement
from mlprogram.interpreters import SequentialProgram
from mlprogram.interpreters import Interpreter


class MockInterpreter(Interpreter):
    def eval(self, code: str) -> int:
        return int(code)

    def eval_references(self, code):
        return {stmt.reference: int(stmt.code) for stmt in code.statements}


class TestTestCaseResult(unittest.TestCase):
    def test_simple_case(self):
        acc = TestCaseResult(MockInterpreter())
        self.assertAlmostEqual(1.0, acc({"ground_truth": ["0"]}, "0"))
        self.assertAlmostEqual(0.0, acc({"ground_truth": ["0"]}, "1"))
        self.assertAlmostEqual(1.0, acc({"ground_truth": ["0", "1"]}, "1"))

    def test_use_input(self):
        acc = TestCaseResult(MockInterpreter(), use_input=True)
        self.assertAlmostEqual(1.0, acc({"input": 0}, "0"))

    def test_reference(self):
        acc = TestCaseResult(MockInterpreter(), reference=True)
        self.assertAlmostEqual(
            1.0,
            acc({"ground_truth":
                 [SequentialProgram([Statement(Reference("0"), "0")])]},
                SequentialProgram([Statement(Reference("0"), "0")]))
        )

    def test_custom_metric(self):
        def metric(input, actual):
            return sum(input["ground_truth"]) + actual

        acc = TestCaseResult(MockInterpreter(), metric=metric)
        self.assertAlmostEqual(0.0, acc({"ground_truth": ["0"]}, "0"))
        self.assertAlmostEqual(1.0, acc({"ground_truth": ["0"]}, "1"))
        self.assertAlmostEqual(2.0, acc({"ground_truth": ["0", "1"]}, "1"))


if __name__ == "__main__":
    unittest.main()
