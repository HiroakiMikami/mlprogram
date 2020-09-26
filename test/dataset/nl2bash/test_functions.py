import unittest

from mlprogram.languages import Token
from mlprogram.datasets.nl2bash import TokenizeQuery, \
    TokenizeToken
from mlprogram.datasets.nl2bash.functions import get_subtokens


class TestGetSubtokens(unittest.TestCase):
    def test_words(self):
        self.assertEqual([Token(None, "foo", "foo")], get_subtokens("foo"))

    def test_numbers(self):
        self.assertEqual(
            [Token(None, "foo", "foo"), Token(None, "10", "10")],
            get_subtokens("foo10"))

    def test_others(self):
        self.assertEqual(
            [Token(None, "/", "/"), Token(None, "foo", "foo")],
            get_subtokens("/foo"))
        self.assertEqual(
            [Token(None, "$", "$"), Token(None, "{", "{"),
             Token(None, "foo", "foo"), Token(None, "10", "10"),
             Token(None, "}", "}")],
            get_subtokens("${" + "foo10" + "}"))


class TestTokenizeAnnotation(unittest.TestCase):
    def test_simple_case(self):
        query = TokenizeQuery()("foo bar")
        self.assertEqual(["foo", "bar"], query.query_for_dnn)
        self.assertEqual([Token(None, "foo", "foo"),
                          Token(None, "bar", "bar")],
                         query.reference)

    def test_subtokens(self):
        query = TokenizeQuery()('foo.bar')
        self.assertEqual(["SUB_START", "foo", ".", "bar", "SUB_END"],
                         query.query_for_dnn)
        self.assertEqual(
            [Token(None, "SUB_START", ""), Token(None, "foo", "foo"),
             Token(None, ".", "."),
             Token(None, "bar", "bar"), Token(None, "SUB_END", "")],
            query.reference)


class TestTokenizeToken(unittest.TestCase):
    def test_simple_case(self):
        self.assertEqual(["test"], TokenizeToken()("test"))

    def test_subtokens(self):
        query = TokenizeToken()('foo.bar')
        self.assertEqual(["foo", ".", "bar"], query)


if __name__ == "__main__":
    unittest.main()
