import unittest
from mlprogram.metrics import TestCaseResult
from mlprogram.interpreters import Interpreter


class MockInterpreter(Interpreter):
    def eval(self, code: str) -> int:
        return int(code)


class TestTestCaseResult(unittest.TestCase):
    def test_simple_case(self):
        acc = TestCaseResult(str, MockInterpreter())
        self.assertAlmostEqual(1.0, acc({"ground_truth": ["0"]}, "0"))
        self.assertAlmostEqual(0.0, acc({"ground_truth": ["0"]}, "1"))
        self.assertAlmostEqual(1.0, acc({"ground_truth": ["0", "1"]}, "1"))

    def test_custom_metric(self):
        def metric(input, actual):
            return sum(input["ground_truth"]) + actual

        acc = TestCaseResult(str, MockInterpreter(), metric)
        self.assertAlmostEqual(0.0, acc({"ground_truth": ["0"]}, "0"))
        self.assertAlmostEqual(1.0, acc({"ground_truth": ["0"]}, "1"))
        self.assertAlmostEqual(2.0, acc({"ground_truth": ["0", "1"]}, "1"))


if __name__ == "__main__":
    unittest.main()
