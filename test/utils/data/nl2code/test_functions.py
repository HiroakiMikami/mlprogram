import unittest
import numpy as np
from torchnlp.encoders import LabelEncoder
from nl2prog.utils import Query
from nl2prog.utils.data import Entry, ListDataset, get_samples
from nl2prog.utils.data.nl2code import to_train_dataset
from nl2prog.language.ast import Node, Leaf, Field
from nl2prog.language.action import ast_to_action_sequence
from nl2prog.encoders import ActionSequenceEncoder


def tokenize(query: str):
    return query.split(" ")


def tokenize_query(query: str):
    return Query(query.split(" "), query.split(" "))


def to_action_sequence(code: str):
    ast = Node("Assign",
               [Field("name", "Name",
                      Node("Name", [Field("id", "str", Leaf("str", "x"))])),
                Field("value", "expr",
                      Node("Op", [
                           Field("op", "str", Leaf("str", "+")),
                           Field("arg0", "expr",
                                 Node("Name", [Field("id", "str",
                                                     Leaf("str", "y"))])),
                           Field("arg1", "expr",
                                 Node("Number", [Field("value", "number",
                                                       Leaf("number", "1"))]))]
                           ))])
    return ast_to_action_sequence(ast, tokenizer=tokenize)


class TestToTrainDataset(unittest.TestCase):
    def test_simple_case(self):
        entries = [Entry("foo bar", "y = x + 1")]
        dataset = ListDataset([entries])
        d = get_samples(dataset, tokenize, to_action_sequence)
        words = ["foo", "bar"]
        qencoder = LabelEncoder(words, 0)
        aencoder = ActionSequenceEncoder(d, 0)
        tdataset = to_train_dataset(dataset, tokenize_query, tokenize,
                                    to_action_sequence, qencoder, aencoder)
        (query_tensor, action_tensor, prev_action_tensor), ground_truth = \
            tdataset[0]
        self.assertTrue(np.array_equal([1, 2], query_tensor.numpy()))
        self.assertTrue(np.array_equal(
            [
                [1, 2, 0], [3, 3, 1], [5, 4, 2], [5, 4, 2], [4, 3, 1],
                [5, 5, 5], [5, 5, 5], [4, 5, 5], [5, 4, 8], [5, 4, 8],
                [4, 5, 5], [8, 6, 11], [8, 6, 11]
            ],
            action_tensor.numpy()
        ))
        self.assertTrue(np.array_equal(
            [
                [2, -1, -1], [3, -1, -1], [4, -1, -1], [-1, 2, -1],
                [-1, 1, -1], [5, -1, -1], [-1, 3, -1], [-1, 1, -1],
                [4, -1, -1], [-1, 4, -1], [-1, 1, -1], [6, -1, -1],
                [-1, 5, -1]
            ],
            prev_action_tensor.numpy()
        ))
        self.assertTrue(np.array_equal(
            [
                [3, -1, -1], [4, -1, -1], [-1, 2, -1], [-1, 1, -1],
                [5, -1, -1], [-1, 3, -1], [-1, 1, -1], [4, -1, -1],
                [-1, 4, -1], [-1, 1, -1], [6, -1, -1], [-1, 5, -1],
                [-1, 1, -1]
            ],
            ground_truth.numpy()
        ))

    def test_impossible_case(self):
        entries = [Entry("foo bar", "y = x + 1")]
        dataset = ListDataset([entries])
        d = get_samples(dataset, tokenize, to_action_sequence)
        words = ["foo", "bar"]
        d.tokens = ["y", "1"]
        qencoder = LabelEncoder(words, 0)
        aencoder = ActionSequenceEncoder(d, 0)
        tdataset = to_train_dataset(dataset, tokenize_query, tokenize,
                                    to_action_sequence, qencoder, aencoder)
        self.assertEqual(0, len(tdataset))


if __name__ == "__main__":
    unittest.main()
