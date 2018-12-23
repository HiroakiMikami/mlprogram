import unittest
from src.utils import TopKElement, bleu4
import os
import numpy as np


class TestUtils(unittest.TestCase):
    def test_top_k_element(self):
        q = TopKElement(2)
        cnt = 0
        q.push(1, lambda: cnt)
        self.assertEqual(q.queue, [[1, 0, 0]])
        q.push(2, lambda: cnt)
        self.assertEqual(q.queue, [[1, 0, 0], [2, 1, 0]])
        q.push(3, lambda: cnt)
        self.assertEqual(q.queue, [[2, 1, 0], [3, 2, 0]])
        q.push(1, lambda: cnt)
        self.assertEqual(q.queue, [[2, 1, 0], [3, 2, 0]])

    def test_bleu4(self):
        self.assertEqual(bleu4("def f():\n  pass\n", "def f():\n  pass\n"), 1)
        self.assertTrue(
            bleu4("def f():\n  pass\n", "def f(arg):\n  pass\n") > 0.4)
        self.assertTrue(
            bleu4("def f():\n  pass\n", "def f(arg):\n  pass\n") < 0.5)


if __name__ == "__main__":
    unittest.main()
