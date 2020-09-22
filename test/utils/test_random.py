import unittest
import numpy as np
from mlprogram.utils.random import split


class TestSplit(unittest.TestCase):
    def test_happy_path(self):
        ns = split(np.random.RandomState(0), 10, 3, 1e-5)
        self.assertEqual(10, sum(ns))


if __name__ == "__main__":
    unittest.main()
