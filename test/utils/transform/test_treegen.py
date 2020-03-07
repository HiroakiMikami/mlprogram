import unittest
import numpy as np
from nl2prog.utils import Query
from nl2prog.utils.data import Entry, ListDataset, get_samples
from nl2prog.language.ast import Node, Field, Leaf
from nl2prog.language.action import ast_to_action_sequence, ActionOptions
from nl2prog.encoders import ActionSequenceEncoder
from nl2prog.utils.transform.treegen import TransformCode


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
    return ast_to_action_sequence(ast,
                                  tokenizer=tokenize,
                                  options=ActionOptions(False, False))


class TestToTrainDataset(unittest.TestCase):
    def test_simple_case(self):
        entries = [Entry("ab test", "y = x + 1")]
        dataset = ListDataset([entries])
        d = get_samples(dataset, tokenize, to_action_sequence)
        aencoder = ActionSequenceEncoder(d, 0)
        transform = TransformCode(to_action_sequence, aencoder, 2,
                                  ActionOptions(False, False))
        prev_query, prev_rule_query, depth, matrix = transform("y = x + 1",
                                                               ["ab", "test"])
        self.assertTrue(np.array_equal(
            [
                [2, -1, -1], [3, -1, -1], [4, -1, -1], [-1, 2, -1],
                [5, -1, -1], [-1, 3, -1], [4, -1, -1], [-1, 4, -1],
                [6, -1, -1]
            ],
            prev_query.numpy()
        ))
        self.assertTrue(np.array_equal(
            [
                # Root -> Root
                [[1, -1, -1], [1, -1, -1], [-1, -1, -1]],
                # Assign -> Name, expr
                [[2, -1, -1], [3, -1, -1], [4, -1, -1]],
                # Name -> str
                [[3, -1, -1], [5, -1, -1], [-1, -1, -1]],
                # str -> "x"
                [[-1, -1, -1], [-1, 2, -1], [-1, -1, -1]],
                # Op -> str, expr, expr
                [[6, -1, -1], [5, -1, -1], [4, -1, -1]],
                # str -> "+"
                [[-1, -1, -1], [-1, 3, -1], [-1, -1, -1]],
                # Name -> str
                [[3, -1, -1], [5, -1, -1], [-1, -1, -1]],
                # str -> "y"
                [[-1, -1, -1], [-1, 4, -1], [-1, -1, -1]],
                # Number -> number
                [[7, -1, -1], [8, -1, -1], [-1, -1, -1]],
            ],
            prev_rule_query.numpy()
        ))
        self.assertTrue(np.array_equal(
            [[0], [1], [2], [3], [2], [3], [3], [4], [3]],
            depth.numpy()
        ))
        self.assertTrue(np.array_equal(
            [[0, 1, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 1, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 1, 1, 0, 1],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0]],
            matrix.numpy()
        ))

    def test_impossible_case(self):
        entries = [Entry("foo bar", "y = x + 1")]
        dataset = ListDataset([entries])
        d = get_samples(dataset, tokenize, to_action_sequence)
        d.tokens = ["y", "1"]
        aencoder = ActionSequenceEncoder(d, 0)
        transform = TransformCode(to_action_sequence, aencoder, 3,
                                  ActionOptions(False, False))
        result = transform("y = x + 1", ["ab", "test"])
        self.assertEqual(None, result)


if __name__ == "__main__":
    unittest.main()
