import unittest
import torch
import numpy as np
from mlprogram.languages.csg import Interpreter
from mlprogram.languages.csg import show
from mlprogram.languages.csg.ast import Rectangle
from mlprogram.languages.csg.transform import TransformCanvas
from mlprogram.languages.csg.transform import AddTestCases

from mlprogram.interpreters import Reference
from mlprogram.interpreters import Statement
from mlprogram.interpreters import SequentialProgram


class TestAddTestCases(unittest.TestCase):
    def test_non_reference(self):
        f = AddTestCases(Interpreter(1, 1, 1))
        result = f({"ground_truth": Rectangle(1, 1)})
        self.assertEqual("#\n",
                         show(result["input"][1]))

    def test_reference(self):
        f = AddTestCases(Interpreter(1, 1, 1), reference=True)
        result = f({
            "ground_truth": SequentialProgram([
                Statement(Reference(0), Rectangle(0, 0)),
                Statement(Reference(1), Rectangle(1, 1))
            ])
        })
        self.assertEqual("#\n", show(result["input"][1]))


class TestTransformCanvas(unittest.TestCase):
    def test_test_case(self):
        f = TransformCanvas(["input"])
        self.assertTrue(np.array_equal(
            torch.tensor([[0.5, -0.5], [-0.5, 0.5]]).reshape(1, 2, 2),
            f({
                "input": (None, np.array([[True, False], [False, True]]))
            })["processed_input"]
        ))

    def test_variables(self):
        f = TransformCanvas(["input", "variables"])
        self.assertTrue(np.array_equal(
            torch.tensor([[0.5, -0.5], [-0.5, 0.5]]).reshape(1, 1, 2, 2),
            f({
                "input": np.array([[True, False], [False, True]]),
                "variables": [np.array([[True, False], [False, True]])]
            })["variables"]
        ))


if __name__ == "__main__":
    unittest.main()
