from mlprogram.datasets.django import SplitValue, TokenizeQuery
from mlprogram.languages import Token


class TestTokenizeQuery(object):
    def test_simple_case(self):
        reference = TokenizeQuery()("foo bar")
        assert [Token(None, "foo", "foo"),
                Token(None, "bar", "bar")] == reference

    def test_quoted_string(self):
        reference = TokenizeQuery()('"quoted string" test')
        assert [Token(None, "####0####", 'quoted string'),
                Token(None, "test", "test")] == reference
        reference = TokenizeQuery()('"quoted string" "quoted string" test')
        assert [Token(None, '####0####', "quoted string"),
                Token(None, "####0####", 'quoted string'),
                Token(None, "test", "test")] == reference

    def test_package_name_like_string(self):
        reference = TokenizeQuery()('foo.bar')
        assert [Token(None, "foo.bar", "foo.bar"),
                Token(None, "foo", "foo"),
                Token(None, "bar", "bar")] == reference


class TestSplitValue(object):
    def test_simple_case(self):
        assert ["test"] == SplitValue()("test")

    def test_camel_case(self):
        assert ["Foo", "Bar"] == SplitValue(True)("FooBar")
