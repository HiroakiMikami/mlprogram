import unittest

from mlprogram.utils import Token
from mlprogram.datasets.django import TokenizeQuery


class TestTokenizeQuery(unittest.TestCase):
    def test_simple_case(self):
        query = TokenizeQuery()("foo bar")
        self.assertEqual([Token(None, "foo"), Token(None, "bar")],
                         query.reference)
        self.assertEqual(["foo", "bar"], query.query_for_dnn)

    def test_quoted_string(self):
        query = TokenizeQuery()('"quoted string" test')
        self.assertEqual([Token(None, 'quoted string'), Token(None, "test")],
                         query.reference)
        self.assertEqual(["####0####", "test"], query.query_for_dnn)
        query = TokenizeQuery()('"quoted string" "quoted string" test')
        self.assertEqual(
            [Token(None, 'quoted string'), Token(None, 'quoted string'),
             Token(None, "test")],
            query.reference)
        self.assertEqual(["####0####", "####0####", "test"],
                         query.query_for_dnn)

    def test_package_name_like_string(self):
        query = TokenizeQuery()('foo.bar')
        self.assertEqual(
            [Token(None, "foo.bar"), Token(None, "foo"), Token(None, "bar")],
            query.reference)
        self.assertEqual(["foo.bar", "foo", "bar"], query.query_for_dnn)


if __name__ == "__main__":
    unittest.main()
