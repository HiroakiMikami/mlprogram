import unittest
import ast
import numpy as np
from nl2prog.utils import Query
from nl2prog.utils.data import Entry, ListDataset
from nl2prog.utils.data.nl2code \
    import get_samples, to_train_dataset, to_eval_dataset
from nl2prog.language.python import to_ast
from nl2prog.language.action import ast_to_action_sequence
from nl2prog.encoders import Encoder


def tokenize(query: str):
    return query.split(" ")


def tokenize_query(query: str):
    return Query(query.split(" "), query.split(" "))


def to_action_sequence(code: str):
    return ast_to_action_sequence(to_ast(ast.parse(code).body[0]),
                                  tokenizer=tokenize)


class TestGetSamples(unittest.TestCase):
    def test_get_samples(self):
        entries = [Entry("foo bar", "y = x + 1"),
                   Entry("test foo", "f(x)")]
        dataset = ListDataset([entries])
        d = get_samples(dataset, tokenize_query, tokenize, to_action_sequence)
        self.assertEqual(["foo", "bar", "test", "foo"], d.words)
        self.assertEqual(["y", "x", "1", "f", "x"], d.tokens)
        self.assertEqual(10, len(d.rules))
        self.assertEqual(24, len(d.node_types))


class TestToTrainDataset(unittest.TestCase):
    def test_simple_case(self):
        entries = [Entry("foo bar", "y = x + 1")]
        dataset = ListDataset([entries])
        d = get_samples(dataset, tokenize_query, tokenize,
                        to_action_sequence)
        d.words = ["foo", "bar"]
        encoder = Encoder(d, 0, 0)
        tdataset = to_train_dataset(dataset, tokenize_query, tokenize,
                                    to_action_sequence, encoder)
        query_tensor, action_tensor, prev_action_tensor = tdataset[0]
        self.assertTrue(np.array_equal([1, 2], query_tensor.numpy()))
        self.assertEqual((13, 3), action_tensor.shape)
        self.assertEqual((14, 3), prev_action_tensor.shape)

    def test_impossible_case(self):
        entries = [Entry("foo bar", "y = x + 1")]
        dataset = ListDataset([entries])
        d = get_samples(dataset, tokenize_query, tokenize,
                        to_action_sequence)
        d.words = ["foo", "bar"]
        d.tokens = ["y", "1"]
        encoder = Encoder(d, 0, 0)
        tdataset = to_train_dataset(dataset, tokenize_query, tokenize,
                                    to_action_sequence, encoder)
        self.assertEqual(0, len(tdataset))


class TestToEvalDataset(unittest.TestCase):
    def test_simple_case(self):
        groups = [[Entry("foo bar", "y = x1")]]
        dataset = ListDataset(groups)
        d = get_samples(dataset, tokenize_query, tokenize,
                        to_action_sequence)
        d.words = ["foo", "bar"]
        encoder = Encoder(d, 0, 0)
        vdataset = to_eval_dataset(dataset, tokenize_query, tokenize,
                                   to_action_sequence, encoder)
        query, ref = vdataset[0]
        self.assertEqual(Query(["foo", "bar"], ["foo", "bar"]), query)
        self.assertEqual(["y = x1"], ref)


if __name__ == "__main__":
    unittest.main()
