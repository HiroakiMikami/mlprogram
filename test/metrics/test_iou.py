import unittest
import numpy as np
from mlprogram.metrics import Iou
from mlprogram.interpreters import Interpreter
from mlprogram.utils import Reference


class MockInterpreter(Interpreter[str, np.array]):
    def eval(self, code: str):
        return np.array([True, True, False], dtype=np.bool)

    def eval_references(self, code):
        return {key: self.eval(code) for key, code in code}


class TestIou(unittest.TestCase):
    def test_simple_case(self):
        iou = Iou(MockInterpreter())
        gt = np.array([False, True, False], dtype=np.bool)
        actual = "code"
        self.assertAlmostEqual(0.5, iou({"test_case": gt}, actual))

    def test_referfence(self):
        iou = Iou(MockInterpreter(), reference=True)
        gt = np.array([False, True, False], dtype=np.bool)
        actual = "code"
        self.assertAlmostEqual(0.5, iou({"test_case": gt},
                                        [(Reference(""), actual)]))


if __name__ == "__main__":
    unittest.main()
