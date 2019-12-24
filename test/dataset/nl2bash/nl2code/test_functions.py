import unittest

from nl2prog.dataset.nl2bash.nl2code import tokenize_query, \
    tokenize_token, get_subtokens


class TestGetSubtokens(unittest.TestCase):
    def test_words(self):
        self.assertEqual(["foo"], get_subtokens("foo"))

    def test_numbers(self):
        self.assertEqual(["foo", "10"], get_subtokens("foo10"))

    def test_others(self):
        self.assertEqual(["/", "foo"], get_subtokens("/foo"))
        self.assertEqual(["$", "{", "foo", "10", "}"],
                         get_subtokens("${" + "foo10" + "}"))


class TestTokenizeAnnotation(unittest.TestCase):
    def test_simple_case(self):
        query = tokenize_query("foo bar")
        self.assertEqual(["foo", "bar"], query.query_for_dnn)
        self.assertEqual(["foo", "bar"], query.query_for_synth)

    def test_subtokens(self):
        query = tokenize_query('foo.bar')
        self.assertEqual(["SUB_START", "foo", ".", "bar", "SUB_END"],
                         query.query_for_dnn)
        self.assertEqual(["SUB_START", "foo", ".", "bar", "SUB_END"],
                         query.query_for_synth)


class TestTokenizeToken(unittest.TestCase):
    def test_simple_case(self):
        self.assertEqual(["test"], tokenize_token("test"))

    def test_subtokens(self):
        query = tokenize_token('foo.bar')
        self.assertEqual(["foo", ".", "bar"], query)


if __name__ == "__main__":
    unittest.main()
