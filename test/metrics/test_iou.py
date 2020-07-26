import unittest
import numpy as np
from mlprogram.metrics import Iou


class TestIou(unittest.TestCase):
    def test_simple_case(self):
        iou = Iou()
        gt = np.array([False, True, False], dtype=np.bool)
        self.assertAlmostEqual(0.5,
                               iou({"ground_truth": [gt]},
                                   np.array([True, True, False], dtype=np.bool)
                                   ))


if __name__ == "__main__":
    unittest.main()
