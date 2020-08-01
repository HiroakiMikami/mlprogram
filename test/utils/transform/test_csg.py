import unittest
import torch
import numpy as np
from mlprogram.utils.transform.csg import TransformCanvas


class TestTransformCanvas(unittest.TestCase):
    def test_test_case(self):
        f = TransformCanvas(["input"])
        self.assertTrue(np.array_equal(
            torch.tensor([[0.5, -0.5], [-0.5, 0.5]]).reshape(1, 2, 2),
            f({
                "input": np.array([[True, False], [False, True]])
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
