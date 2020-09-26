import unittest

from mlprogram.languages import Token
from mlprogram.datasets.django import TokenizeQuery, TokenizeToken


class TestTokenizeQuery(unittest.TestCase):
    def test_simple_case(self):
        query = TokenizeQuery()("foo bar")
        self.assertEqual([Token(None, "foo", "foo"),
                          Token(None, "bar", "bar")],
                         query.reference)
        self.assertEqual(["foo", "bar"], query.query_for_dnn)

    def test_quoted_string(self):
        query = TokenizeQuery()('"quoted string" test')
        self.assertEqual([Token(None, "####0####", 'quoted string'),
                          Token(None, "test", "test")],
                         query.reference)
        self.assertEqual(["####0####", "test"], query.query_for_dnn)
        query = TokenizeQuery()('"quoted string" "quoted string" test')
        self.assertEqual(
            [Token(None, '####0####', "quoted string"),
             Token(None, "####0####", 'quoted string'),
             Token(None, "test", "test")],
            query.reference)
        self.assertEqual(["####0####", "####0####", "test"],
                         query.query_for_dnn)

    def test_package_name_like_string(self):
        query = TokenizeQuery()('foo.bar')
        self.assertEqual(
            [Token(None, "foo.bar", "foo.bar"),
             Token(None, "foo", "foo"), Token(None, "bar", "bar")],
            query.reference)
        self.assertEqual(["foo.bar", "foo", "bar"], query.query_for_dnn)


class TestTokenizeToken(unittest.TestCase):
    def test_simple_case(self):
        self.assertEqual(["test"], TokenizeToken()("test"))

    def test_camel_case(self):
        self.assertEqual(["Foo", "Bar"], TokenizeToken(True)("FooBar"))


if __name__ == "__main__":
    unittest.main()
