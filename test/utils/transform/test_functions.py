import unittest
import numpy as np

from nl2prog.utils import Query
from nl2prog.utils.data import Entry, ListDataset, get_samples
from nl2prog.language.ast import Node, Leaf, Field
from nl2prog.language.action import ast_to_action_sequence
from nl2prog.encoders import ActionSequenceEncoder
from nl2prog.utils.transform \
    import TransformGroundTruth, TransformDataset


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


class TestTransformGroundTruth(unittest.TestCase):
    def test_simple_case(self):
        entries = [Entry("foo bar", "y = x + 1")]
        dataset = ListDataset([entries])
        d = get_samples(dataset, tokenize, to_action_sequence)
        aencoder = ActionSequenceEncoder(d, 0)
        transform = TransformGroundTruth(to_action_sequence, aencoder)
        ground_truth = transform("y = x + 1", ["foo", "bar"])
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
        d.tokens = ["y", "1"]
        aencoder = ActionSequenceEncoder(d, 0)
        transform = TransformGroundTruth(to_action_sequence, aencoder)
        ground_truth = transform("y = x + 1", ["foo", "bar"])
        self.assertEqual(None, ground_truth)


class TestTransformDataset(unittest.TestCase):
    def test_happy_path(self):
        dataset = ListDataset([[Entry("foo bar", "y = x + 1")]])
        transform = TransformDataset(lambda x: ([x], 0), lambda x, y: (x, y),
                                     lambda x, y: y)
        dataset = transform(dataset)
        self.assertEqual(1, len(dataset))
        input, action_seq, query, ground_truth = dataset[0]
        self.assertEqual(0, input)
        self.assertEqual("y = x + 1", action_seq)
        self.assertEqual(["foo bar"], query)
        self.assertEqual(["foo bar"], ground_truth)


if __name__ == "__main__":
    unittest.main()
