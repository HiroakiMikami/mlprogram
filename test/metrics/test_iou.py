import unittest
import numpy as np
from mlprogram.metrics import Iou


class TestIou(unittest.TestCase):
    def test_scimple_case(self):
        iou = Iou()
        gts = [np.array([False, True, True], dtype=np.bool),
               np.array([False, True, False], dtype=np.bool)]
        actual = [np.array([True, True, False], dtype=np.bool)]
        self.assertAlmostEqual(0.5, iou({"ground_truth": gts}, actual))


if __name__ == "__main__":
    unittest.main()
