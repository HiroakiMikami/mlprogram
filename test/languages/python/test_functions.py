import unittest

from mlprogram.languages.python import TokenizeToken


class TestTokenizeToken(unittest.TestCase):
    def test_simple_case(self):
        self.assertEqual(["test"], TokenizeToken()("test"))

    def test_camel_case(self):
        self.assertEqual(["Foo", "Bar"], TokenizeToken(True)("FooBar"))


if __name__ == "__main__":
    unittest.main()
