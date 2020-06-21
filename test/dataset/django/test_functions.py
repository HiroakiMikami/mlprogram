import unittest

from mlprogram.utils import Token
from mlprogram.datasets.django import tokenize_query


class TestTokenizeQuery(unittest.TestCase):
    def test_simple_case(self):
        query = tokenize_query("foo bar")
        self.assertEqual([Token(None, "foo"), Token(None, "bar")],
                         query.reference)
        self.assertEqual(["foo", "bar"], query.query_for_dnn)

    def test_quoted_string(self):
        query = tokenize_query('"quoted string" test')
        self.assertEqual([Token(None, 'quoted string'), Token(None, "test")],
                         query.reference)
        self.assertEqual(["####0####", "test"], query.query_for_dnn)
        query = tokenize_query('"quoted string" "quoted string" test')
        self.assertEqual(
            [Token(None, 'quoted string'), Token(None, 'quoted string'),
             Token(None, "test")],
            query.reference)
        self.assertEqual(["####0####", "####0####", "test"],
                         query.query_for_dnn)

    def test_package_name_like_string(self):
        query = tokenize_query('foo.bar')
        self.assertEqual(
            [Token(None, "foo.bar"), Token(None, "foo"), Token(None, "bar")],
            query.reference)
        self.assertEqual(["foo.bar", "foo", "bar"], query.query_for_dnn)


if __name__ == "__main__":
    unittest.main()
