import unittest
import torch
import numpy as np
from mlprogram.utils import Reference
from mlprogram.utils.transform.csg import TransformCanvas


class TestEvaluateGroundTruth(unittest.TestCase):
    def test_test_case(self):
        f = TransformCanvas()
        self.assertTrue(np.array_equal(
            torch.tensor([[0.5, -0.5], [-0.5, 0.5]]).reshape(1, 2, 2),
            f({
                "test_case": np.array([[True, False], [False, True]])
            })["input"]
        ))

    def test_variables(self):
        f = TransformCanvas()
        self.assertTrue(np.array_equal(
            torch.tensor([[0.5, -0.5], [-0.5, 0.5]]).reshape(1, 2, 2),
            f({"variables": {
                Reference(0): np.array([[True, False], [False, True]])
            }})["variables"][Reference(0)]
        ))


if __name__ == "__main__":
    unittest.main()
