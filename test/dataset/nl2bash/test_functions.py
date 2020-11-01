from mlprogram.datasets.nl2bash import SplitValue, TokenizeQuery
from mlprogram.datasets.nl2bash.functions import get_subtokens
from mlprogram.languages import Token


class TestGetSubtokens(object):
    def test_words(self):
        assert [Token(None, "foo", "foo")] == get_subtokens("foo")

    def test_numbers(self):
        assert [Token(None, "foo", "foo"),
                Token(None, "10", "10")] == get_subtokens("foo10")

    def test_others(self):
        assert [Token(None, "/", "/"),
                Token(None, "foo", "foo")] == get_subtokens("/foo")
        assert [Token(None, "$", "$"), Token(None, "{", "{"),
                Token(None, "foo", "foo"), Token(None, "10", "10"),
                Token(None, "}", "}")] == get_subtokens("${" + "foo10" + "}")


class TestTokenizeAnnotation(object):
    def test_simple_case(self):
        reference = TokenizeQuery()("foo bar")
        assert [Token(None, "foo", "foo"),
                Token(None, "bar", "bar")] == reference

    def test_subtokens(self):
        reference = TokenizeQuery()('foo.bar')
        assert [Token(None, "SUB_START", ""),
                Token(None, "foo", "foo"),
                Token(None, ".", "."),
                Token(None, "bar", "bar"),
                Token(None, "SUB_END", "")] == reference


class TestSplitValue(object):
    def test_simple_case(self):
        assert ["test"] == SplitValue()("test")

    def test_subtokens(self):
        tokens = SplitValue()("foo.bar")
        assert ["foo", ".", "bar"] == tokens
