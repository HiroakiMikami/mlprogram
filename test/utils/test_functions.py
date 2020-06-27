import unittest
from mlprogram.utils import Compose, Sequence


class TestCompose(unittest.TestCase):
    def test_happy_path(self):
        f = Compose(lambda x: x + 1, lambda y: y * 2)
        self.assertEqual(6, f(2))

    def test_value_is_none(self):
        f = Compose(lambda x: x + 1, lambda y: y * 2)
        self.assertEqual(None, f(None))

    def test_f_return_none(self):
        f = Compose(lambda x: None, lambda y: y * 2)
        self.assertEqual(None, f(2))


class TestSequence(unittest.TestCase):
    def test_happy_path(self):
        f = Sequence(f0=lambda x: {"x": x["x"] + 1},
                     f1=lambda x: {"x": x["x"] * 2})
        self.assertEqual({"x": 6}, f({"x": 2}))

    def test_f_return_none(self):
        f = Sequence(f0=lambda x: None, f1=lambda x: {"x": x["x"] * 2})
        self.assertEqual(None, f({"x": 2}))


if __name__ == "__main__":
    unittest.main()
