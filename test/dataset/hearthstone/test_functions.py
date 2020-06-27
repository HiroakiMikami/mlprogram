import unittest
from mlprogram.utils import Token
from mlprogram.datasets.hearthstone import tokenize_query


class TestTokenizeQuery(unittest.TestCase):
    def test_happy_path(self):
        query = tokenize_query("w0 w1 NAME_END 1 ATK_END NIL")
        self.assertEqual([Token(None, "w0 w1"), Token(None, "NAME_END"),
                          Token(None, "1"), Token(None, "ATK_END"),
                          Token(None, "NIL")],
                         query.reference)
        self.assertEqual(["w0 w1", "NAME_END", "1", "ATK_END", "NIL"],
                         query.query_for_dnn)


if __name__ == "__main__":
    unittest.main()
