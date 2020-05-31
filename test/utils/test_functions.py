import unittest
from mlprogram.utils import Compose


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


if __name__ == "__main__":
    unittest.main()
