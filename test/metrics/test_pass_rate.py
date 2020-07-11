import unittest
from mlprogram.metrics import TestPassRate
from mlprogram.interpreters import Interpreter


class MockInterpreter(Interpreter):
    def eval(self, code: str) -> int:
        return int(code)


class TestTestPassRate(unittest.TestCase):
    def test_simple_case(self):
        acc = TestPassRate(str, MockInterpreter())
        self.assertAlmostEqual(1.0, acc({"ground_truth": ["0"]}, "0"))
        self.assertAlmostEqual(0.0, acc({"ground_truth": ["0"]}, "1"))
        self.assertAlmostEqual(1.0, acc({"ground_truth": ["0", "1"]}, "1"))


if __name__ == "__main__":
    unittest.main()
