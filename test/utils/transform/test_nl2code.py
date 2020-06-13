import unittest
import numpy as np
from torchnlp.encoders import LabelEncoder
from mlprogram.utils import Query
from mlprogram.utils.data import Entry, ListDataset, get_samples
from mlprogram.asts import Node, Leaf, Field
from mlprogram.encoders import ActionSequenceEncoder
from mlprogram.utils.transform import AstToSingleActionSequence
from mlprogram.utils.transform import TransformCode
from mlprogram.utils.transform.nl2code \
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
    return AstToSingleActionSequence(tokenize=tokenize)(ast)


class TestTransformQuery(unittest.TestCase):
    def test_happy_path(self):
        def tokenize_query(value: str):
            return Query([value], [value + "dnn"])

        transform = TransformQuery(tokenize_query, LabelEncoder(["dnn"]))
        query_for_synth, query_tensor = transform("")
        self.assertEqual([""], query_for_synth)
        self.assertEqual([1], query_tensor["word_nl_query"].numpy().tolist())


class TestTransformEvaluator(unittest.TestCase):
    def test_simple_case(self):
        entries = [Entry("foo bar", "y = x + 1")]
        dataset = ListDataset([entries])
        d = get_samples(dataset, tokenize, to_action_sequence)
        aencoder = ActionSequenceEncoder(d, 0)
        transform = TransformEvaluator(aencoder)
        evaluator = TransformCode(to_action_sequence)("y = x + 1")
        result = transform(evaluator, ["foo", "bar"])
        action_tensor = result["actions"]
        prev_action_tensor = result["previous_actions"]
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

    def test_eval(self):
        entries = [Entry("foo bar", "y = x + 1")]
        dataset = ListDataset([entries])
        d = get_samples(dataset, tokenize, to_action_sequence)
        aencoder = ActionSequenceEncoder(d, 0)
        evaluator = TransformCode(to_action_sequence)("y = x + 1")
        transform = TransformEvaluator(aencoder, train=False)
        result = transform(evaluator, ["foo", "bar"])
        action_tensor = result["actions"]
        prev_action_tensor = result["previous_actions"]

        self.assertTrue(np.array_equal(
            [[-1, -1, -1]],
            action_tensor.numpy()
        ))
        self.assertTrue(np.array_equal(
            [[-1, 1, -1]],
            prev_action_tensor.numpy()
        ))

    def test_impossible_case(self):
        entries = [Entry("foo bar", "y = x + 1")]
        dataset = ListDataset([entries])
        d = get_samples(dataset, tokenize, to_action_sequence)
        d.tokens = ["y", "1"]
        aencoder = ActionSequenceEncoder(d, 0)
        transform = TransformEvaluator(aencoder)
        evaluator = TransformCode(to_action_sequence)("y = x + 1")
        result = transform(evaluator, ["foo", "bar"])
        self.assertEqual(None, result)


if __name__ == "__main__":
    unittest.main()
