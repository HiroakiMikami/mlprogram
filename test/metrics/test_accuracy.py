import unittest
from mlprogram.metrics import Accuracy


class TestAccuracy(unittest.TestCase):
    def test_simple_case(self):
        acc = Accuracy()
        self.assertAlmostEqual(1.0, acc({"ground_truth": ["str"]}, "str"))


if __name__ == "__main__":
    unittest.main()
