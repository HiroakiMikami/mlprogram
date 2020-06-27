import unittest

from mlprogram.utils.python import tokenize_token


class TestTokenizeToken(unittest.TestCase):
    def test_simple_case(self):
        self.assertEqual(["test"], tokenize_token("test"))

    def test_camel_case(self):
        self.assertEqual(["Foo", "Bar"], tokenize_token("FooBar", True))


if __name__ == "__main__":
    unittest.main()
