from mlprogram.datasets.hearthstone import TokenizeQuery
from mlprogram.languages import Token


class TestTokenizeQuery(object):
    def test_happy_path(self):
        reference = TokenizeQuery()("w0 w1 NAME_END 1 ATK_END NIL")
        assert [Token(None, "w0 w1", "w0 w1"),
                Token(None, "NAME_END", "NAME_END"),
                Token(None, "1", "1"),
                Token(None, "ATK_END", "ATK_END"),
                Token(None, "NIL", "NIL")] == reference
