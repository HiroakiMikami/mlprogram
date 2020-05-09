import unittest
from mlprogram.metrics.python import Bleu


class TestBleu(unittest.TestCase):
    def test_bleu(self):
        bleu = Bleu(str, str)
        self.assertEqual(bleu(["def f():\n  pass\n"], "def f():\n  pass\n"), 1)
        self.assertTrue(
            bleu(["def f():\n  pass\n"], "def f(arg):\n  pass\n") > 0.9)


if __name__ == "__main__":
    unittest.main()
