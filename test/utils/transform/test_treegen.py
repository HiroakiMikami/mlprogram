import unittest
import numpy as np
from torchnlp.encoders import LabelEncoder
from mlprogram.utils import Query
from mlprogram.utils.data import Entry, ListDataset, get_samples
from mlprogram.ast import Node, Field, Leaf
from mlprogram.action.action import ast_to_action_sequence, ActionOptions
from mlprogram.encoders import ActionSequenceEncoder
from mlprogram.utils.transform import TransformCode
from mlprogram.utils.transform.treegen \
    import TransformQuery, TransformEvaluator


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


class TestTransformQuery(unittest.TestCase):
    def test_simple_case(self):
        words = ["ab", "test"]
        qencoder = LabelEncoder(words, 0)
        cencoder = LabelEncoder(["a", "b", "t", "e"], 0)
        transform = TransformQuery(tokenize_query, qencoder, cencoder, 3)
        query_for_synth, (word_query, char_query) = transform("ab test")
        self.assertEqual(["ab", "test"], query_for_synth)
        self.assertTrue(np.array_equal([1, 2], word_query.numpy()))
        self.assertTrue(np.array_equal([[1, 2, -1], [3, 4, 0]],
                                       char_query.numpy()))


class TestTransformEvaluator(unittest.TestCase):
    def test_simple_case(self):
        entries = [Entry("ab test", "y = x + 1")]
        dataset = ListDataset([entries])
        d = get_samples(dataset, tokenize, to_action_sequence)
        aencoder = ActionSequenceEncoder(d, 0)
        evaluator = \
            TransformCode(to_action_sequence)("y = x + 1")
        transform = TransformEvaluator(aencoder, 2, 3)
        (prev_action, prev_rule_action, depth, matrix), query = \
            transform(evaluator, ["ab", "test"])
        self.assertTrue(np.array_equal(
            [
                [1, -1, -1], [2, -1, -1], [3, -1, -1], [-1, 1, -1],
                [4, -1, -1], [-1, 2, -1], [3, -1, -1], [-1, 3, -1],
                [5, -1, -1]
            ],
            prev_action.numpy()
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
                [[-1, -1, -1], [-1, 1, -1], [-1, -1, -1]],
                # Op -> str, expr, expr
                [[6, -1, -1], [5, -1, -1], [4, -1, -1]],
                # str -> "+"
                [[-1, -1, -1], [-1, 2, -1], [-1, -1, -1]],
                # Name -> str
                [[3, -1, -1], [5, -1, -1], [-1, -1, -1]],
                # str -> "y"
                [[-1, -1, -1], [-1, 3, -1], [-1, -1, -1]],
                # Number -> number
                [[7, -1, -1], [8, -1, -1], [-1, -1, -1]],
            ],
            prev_rule_action.numpy()
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
        self.assertTrue(np.array_equal(
            [
                [-1, -1, -1], [1, -1, -1], [2, 1, -1], [3, 2, 1],
                [2, 1, -1], [4, 2, 1], [4, 2, 1], [3, 4, 2],
                [4, 2, 1]
            ],
            query.numpy()
        ))

    def test_eval(self):
        entries = [Entry("ab test", "y = x + 1")]
        dataset = ListDataset([entries])
        d = get_samples(dataset, tokenize, to_action_sequence)
        aencoder = ActionSequenceEncoder(d, 0)
        evaluator = \
            TransformCode(to_action_sequence)("y = x + 1")
        transform = TransformEvaluator(aencoder, 2, 3, train=False)
        (prev_action, prev_rule_action, depth, matrix), query = \
            transform(evaluator, ["ab", "test"])
        self.assertTrue(np.array_equal(
            [
                [1, -1, -1], [2, -1, -1], [3, -1, -1], [-1, 1, -1],
                [4, -1, -1], [-1, 2, -1], [3, -1, -1], [-1, 3, -1],
                [5, -1, -1], [-1, 4, -1]
            ],
            prev_action.numpy()
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
                [[-1, -1, -1], [-1, 1, -1], [-1, -1, -1]],
                # Op -> str, expr, expr
                [[6, -1, -1], [5, -1, -1], [4, -1, -1]],
                # str -> "+"
                [[-1, -1, -1], [-1, 2, -1], [-1, -1, -1]],
                # Name -> str
                [[3, -1, -1], [5, -1, -1], [-1, -1, -1]],
                # str -> "y"
                [[-1, -1, -1], [-1, 3, -1], [-1, -1, -1]],
                # Number -> number
                [[7, -1, -1], [8, -1, -1], [-1, -1, -1]],
                [[-1, -1, -1], [-1, 4, -1], [-1, -1, -1]],
            ],
            prev_rule_action.numpy()
        ))
        self.assertTrue(np.array_equal(
            [[0], [1], [2], [3], [2], [3], [3], [4], [3], [4]],
            depth.numpy()
        ))
        self.assertTrue(np.array_equal(
            [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 1, 1, 0, 1, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
            matrix.numpy()
        ))
        self.assertTrue(np.array_equal(
            [
                [-1, -1, -1], [1, -1, -1], [2, 1, -1], [3, 2, 1],
                [2, 1, -1], [4, 2, 1], [4, 2, 1], [3, 4, 2],
                [4, 2, 1], [5, 4, 2]
            ],
            query.numpy()
        ))

    def test_impossible_case(self):
        entries = [Entry("foo bar", "y = x + 1")]
        dataset = ListDataset([entries])
        d = get_samples(dataset, tokenize, to_action_sequence)
        d.tokens = ["y", "1"]
        aencoder = ActionSequenceEncoder(d, 0)
        evaluator = \
            TransformCode(to_action_sequence)("y = x + 1")
        transform = TransformEvaluator(aencoder, 3, 3)
        result = transform(evaluator, ["ab", "test"])
        self.assertEqual(None, result)


if __name__ == "__main__":
    unittest.main()
