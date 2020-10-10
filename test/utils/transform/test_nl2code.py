import numpy as np
from typing import List
from torchnlp.encoders import LabelEncoder
from mlprogram.utils.data import ListDataset, get_samples
from mlprogram.languages import Node, Leaf, Field
from mlprogram.languages import Token
from mlprogram.languages import Parser
from mlprogram.encoders import ActionSequenceEncoder
from mlprogram.utils.transform.action_sequence import TransformCode
from mlprogram.utils.transform.nl2code \
    import TransformQuery, TransformActionSequence


def tokenize(str: str) -> List[Token]:
    return list(map(lambda x: Token(None, x, x), str.split(" ")))


class MockParser(Parser[str]):
    def parse(self, code: str):
        ast = Node("Assign",
                   [Field("name", "Name",
                          Node("Name", [Field("id", "str",
                                              [Leaf("str", "x")])])),
                    Field("value", "expr",
                          Node("Op", [
                              Field("op", "str", [Leaf("str", "+")]),
                              Field("arg0", "expr",
                                    Node("Name", [Field("id", "str",
                                                        [Leaf("str", "y")])])),
                              Field("arg1", "expr",
                                    Node("Number", [
                                        Field("value", "number",
                                              [Leaf("number", "1")])
                                    ]))]
                               ))])
        return ast


class TestTransformQuery(object):
    def test_happy_path(self):
        def tokenize(value: str):
            return [Token(None, value + "dnn", value)]

        transform = TransformQuery(tokenize, LabelEncoder(["dnn"]))
        result = transform({"input": ""})
        assert [Token(None, "dnn", "")] == result["reference"]
        assert [1] == result["word_nl_query"].numpy().tolist()


class TestTransformActionSequence(object):
    def test_simple_case(self):
        entries = [{"input": "foo bar", "ground_truth": "y = x + 1"}]
        dataset = ListDataset(entries)
        d = get_samples(dataset, MockParser())
        aencoder = ActionSequenceEncoder(d, 0)
        transform = TransformActionSequence(aencoder)
        action_sequence = TransformCode(MockParser())({
            "ground_truth": "y = x + 1"
        })["action_sequence"]
        result = transform({
            "action_sequence": action_sequence,
            "reference": [Token(None, "foo", "foo"), Token(None, "bar", "bar")]
        })
        action_tensor = result["actions"]
        prev_action_tensor = result["previous_actions"]
        assert np.array_equal(
            [
                [2, 2, 0], [4, 3, 1], [6, 4, 2], [6, 4, 2], [5, 3, 1],
                [6, 5, 5], [6, 5, 5], [5, 5, 5], [6, 4, 8], [6, 4, 8],
                [5, 5, 5], [9, 6, 11], [9, 6, 11]
            ],
            action_tensor.numpy()
        )
        assert np.array_equal(
            [
                [2, -1, -1], [3, -1, -1], [4, -1, -1], [-1, 1, -1],
                [1, -1, -1], [5, -1, -1], [-1, 2, -1], [1, -1, -1],
                [4, -1, -1], [-1, 3, -1], [1, -1, -1], [6, -1, -1],
                [-1, 4, -1]
            ],
            prev_action_tensor.numpy()
        )

    def test_eval(self):
        entries = [{"input": "foo bar", "ground_truth": "y = x + 1"}]
        dataset = ListDataset(entries)
        d = get_samples(dataset, MockParser())
        aencoder = ActionSequenceEncoder(d, 0)
        action_sequence = TransformCode(MockParser())({
            "ground_truth": "y = x + 1"
        })["action_sequence"]
        transform = TransformActionSequence(aencoder, train=False)
        result = transform({
            "action_sequence": action_sequence,
            "reference": [Token(None, "foo", "foo"), Token(None, "bar", "bar")]
        })
        action_tensor = result["actions"]
        prev_action_tensor = result["previous_actions"]

        assert np.array_equal(
            [[-1, -1, -1]],
            action_tensor.numpy()
        )
        assert np.array_equal(
            [[1, -1, -1]],
            prev_action_tensor.numpy()
        )

    def test_impossible_case(self):
        entries = [{"input": "foo bar", "ground_truth": "y = x + 1"}]
        dataset = ListDataset(entries)
        d = get_samples(dataset, MockParser())
        d.tokens = [("", "y"), ("", "1")]
        aencoder = ActionSequenceEncoder(d, 0)
        transform = TransformActionSequence(aencoder)
        action_sequence = TransformCode(MockParser())({
            "ground_truth": "y = x + 1"
        })["action_sequence"]
        result = transform({
            "action_sequence": action_sequence,
            "reference": [Token(None, "foo", "foo"), Token(None, "bar", "bar")]
        })
        assert result is None
