import unittest

from mlprogram.languages import Token
from mlprogram.datasets.django import TokenizeQuery
from mlprogram.datasets.django import SplitToken


class TestTokenizeQuery(unittest.TestCase):
    def test_simple_case(self):
        reference = TokenizeQuery()("foo bar")
        self.assertEqual([Token(None, "foo", "foo"),
                          Token(None, "bar", "bar")],
                         reference)

    def test_quoted_string(self):
        reference = TokenizeQuery()('"quoted string" test')
        self.assertEqual([Token(None, "####0####", 'quoted string'),
                          Token(None, "test", "test")],
                         reference)
        reference = TokenizeQuery()('"quoted string" "quoted string" test')
        self.assertEqual(
            [Token(None, '####0####', "quoted string"),
             Token(None, "####0####", 'quoted string'),
             Token(None, "test", "test")],
            reference)

    def test_package_name_like_string(self):
        reference = TokenizeQuery()('foo.bar')
        self.assertEqual(
            [Token(None, "foo.bar", "foo.bar"),
             Token(None, "foo", "foo"), Token(None, "bar", "bar")],
            reference)


class TestSplitToken(unittest.TestCase):
    def test_simple_case(self):
        self.assertEqual([Token(None, "test", "test")],
                         SplitToken()(Token(None, "test", "test")))

    def test_camel_case(self):
        self.assertEqual([Token(None, "Foo", "Foo"),
                          Token(None, "Bar", "Bar")],
                         SplitToken(True)(Token(None, "FooBar", "FooBar")))

    def mapped_value(self):
        self.assertEqual([Token(None, "FooBar", "Foo.Bar")],
                         SplitToken(True)(Token(None, "FooBar", "Foo.Bar")))


if __name__ == "__main__":
    unittest.main()
