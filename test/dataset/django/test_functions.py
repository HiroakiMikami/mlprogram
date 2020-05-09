import unittest

from mlprogram.dataset.django import tokenize_query


class TestTokenizeQuery(unittest.TestCase):
    def test_simple_case(self):
        query = tokenize_query("foo bar")
        self.assertEqual(["foo", "bar"], query.query_for_synth)
        self.assertEqual(["foo", "bar"], query.query_for_dnn)

    def test_quoted_string(self):
        query = tokenize_query('"quoted string" test')
        self.assertEqual(['quoted string', "test"], query.query_for_synth)
        self.assertEqual(["####0####", "test"], query.query_for_dnn)
        query = tokenize_query('"quoted string" "quoted string" test')
        self.assertEqual(['quoted string', 'quoted string', "test"],
                         query.query_for_synth)
        self.assertEqual(["####0####", "####0####", "test"],
                         query.query_for_dnn)

    def test_package_name_like_string(self):
        query = tokenize_query('foo.bar')
        self.assertEqual(["foo.bar", "foo", "bar"], query.query_for_synth)
        self.assertEqual(["foo.bar", "foo", "bar"], query.query_for_dnn)


if __name__ == "__main__":
    unittest.main()
