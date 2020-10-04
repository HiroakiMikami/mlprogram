import unittest
from mlprogram.utils.transform import NormalizeGroudTruth


class TestNormalizeGroundTruth(unittest.TestCase):
    def test_happy_path(self):
        f = NormalizeGroudTruth(lambda x: len(x))
        self.assertEqual(1, f({"ground_truth": [1]})["ground_truth"])

    def test_return_None(self):
        f = NormalizeGroudTruth(lambda x: None)
        self.assertEqual([1], f({"ground_truth": [1]})["ground_truth"])


if __name__ == "__main__":
    unittest.main()
