import unittest
from mlprogram.languages.python.metrics import Bleu


class TestBleu(unittest.TestCase):
    def test_bleu(self):
        bleu = Bleu()
        self.assertEqual(
            bleu({"ground_truth": "def f():\n  pass\n"},
                 "def f():\n  pass\n"), 1)
        self.assertTrue(
            bleu({"ground_truth": "def f():\n  pass\n"},
                 "def f(arg):\n  pass\n") > 0.9)


if __name__ == "__main__":
    unittest.main()
