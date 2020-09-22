import unittest

from mlprogram.nn import Function


class TestPick(unittest.TestCase):
    def test_parameters(self):
        f = Function(lambda x: x)
        params = dict(f.named_parameters())
        self.assertEqual(0, len(params))

    def test_happy_path(self):
        f = Function(lambda x: x)
        out = f(10)
        self.assertEqual(10, out)


if __name__ == "__main__":
    unittest.main()
