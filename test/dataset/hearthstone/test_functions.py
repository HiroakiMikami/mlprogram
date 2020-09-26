import unittest
from mlprogram.languages import Token
from mlprogram.datasets.hearthstone import TokenizeQuery


class TestTokenizeQuery(unittest.TestCase):
    def test_happy_path(self):
        query = TokenizeQuery()("w0 w1 NAME_END 1 ATK_END NIL")
        self.assertEqual([Token(None, "w0 w1", "w0 w1"),
                          Token(None, "NAME_END", "NAME_END"),
                          Token(None, "1", "1"),
                          Token(None, "ATK_END", "ATK_END"),
                          Token(None, "NIL", "NIL")],
                         query.reference)
        self.assertEqual(["w0 w1", "NAME_END", "1", "ATK_END", "NIL"],
                         query.query_for_dnn)


if __name__ == "__main__":
    unittest.main()
