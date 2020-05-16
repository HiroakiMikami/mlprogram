import unittest
import ast
from mlprogram.utils import Query
from mlprogram.language.python import to_ast
from mlprogram.action.action import ast_to_action_sequence, ActionOptions
from mlprogram.utils.data \
    import Entry, ListDataset, to_eval_dataset, get_samples, get_words, \
    get_characters


def tokenize(query: str):
    return query.split(" ")


def tokenize_query(query: str):
    return Query(query.split(" "), query.split(" "))


def to_action_sequence(code: str):
    return ast_to_action_sequence(to_ast(ast.parse(code).body[0]),
                                  tokenizer=tokenize)


class TestGetWords(unittest.TestCase):
    def test_get_words(self):
        entries = [Entry("foo bar", "y = x + 1"),
                   Entry("test foo", "f(x)")]
        dataset = ListDataset([entries])
        words = get_words(dataset, tokenize_query)
        self.assertEqual(["foo", "bar", "test", "foo"], words)


class TestGetCharacters(unittest.TestCase):
    def test_get_characters(self):
        entries = [Entry("foo bar", "y = x + 1"),
                   Entry("test foo", "f(x)")]
        dataset = ListDataset([entries])
        chars = get_characters(dataset, tokenize_query)
        self.assertEqual([
            "f", "o", "o",
            "b", "a", "r",
            "t", "e", "s", "t",
            "f", "o", "o"], chars)


class TestGetSamples(unittest.TestCase):
    def test_get_samples(self):
        entries = [Entry("foo bar", "y = x + 1"),
                   Entry("test foo", "f(x)")]
        dataset = ListDataset([entries])
        d = get_samples(dataset, tokenize, to_action_sequence)
        self.assertEqual(["y", "x", "1", "f", "x"], d.tokens)
        self.assertEqual(12, len(d.rules))
        self.assertEqual(28, len(d.node_types))
        self.assertEqual(ActionOptions(True, True), d.options)


class TestToEvalDataset(unittest.TestCase):
    def test_simple_case(self):
        groups = [[Entry("foo bar", "y = x1")]]
        dataset = ListDataset(groups)
        vdataset = to_eval_dataset(dataset)
        query, ref = vdataset[0]
        self.assertEqual("foo bar", query)
        self.assertEqual(["y = x1"], ref)


if __name__ == "__main__":
    unittest.main()
