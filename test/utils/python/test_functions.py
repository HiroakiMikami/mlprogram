import unittest

from mlprogram.utils import Token
from mlprogram.utils.python import tokenize_query, tokenize_token


class TestTokenizeQuery(unittest.TestCase):
    def test_simple_case(self):
        query = tokenize_query("foo bar")
        self.assertEqual([Token(None, "foo"), Token(None, "bar")],
                         query.reference)
        self.assertEqual(["foo", "bar"], query.query_for_dnn)

    def test_package_name_like_string(self):
        query = tokenize_query('foo.bar')
        self.assertEqual(
            [Token(None, "foo.bar"), Token(None, "foo"), Token(None, "bar")],
            query.reference)
        self.assertEqual(["foo.bar", "foo", "bar"], query.query_for_dnn)


class TestTokenizeToken(unittest.TestCase):
    def test_simple_case(self):
        self.assertEqual(["test"], tokenize_token("test"))

    def test_camel_case(self):
        self.assertEqual(["Foo", "Bar"], tokenize_token("FooBar", True))


if __name__ == "__main__":
    unittest.main()
