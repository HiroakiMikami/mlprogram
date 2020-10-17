import numpy as np

from mlprogram import Environment
from mlprogram.utils.data import ListDataset, get_samples
from mlprogram.languages import Node, Leaf, Field
from mlprogram.languages import Token
from mlprogram.languages import Parser
from mlprogram.encoders import ActionSequenceEncoder
from mlprogram.utils.transform.action_sequence \
    import TransformCode, TransformGroundTruth, \
    TransformActionSequenceForRnnDecoder


def tokenize(query: str):
    return list(map(lambda x: Token(None, x, x), query.split(" ")))


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


class TestTransformCode(object):
    def test_simple_case(self):
        transform = TransformCode(MockParser())
        action_sequence = transform(Environment(
            supervisions={"ground_truth": "y = x + 1"}
        )).supervisions["action_sequence"]
        assert action_sequence.head is None


class TestTransformGroundTruth(object):
    def test_simple_case(self):
        entries = [Environment(
            inputs={"input": "foo bar"},
            supervisions={"ground_truth": "y = x + 1"}
        )]
        dataset = ListDataset(entries)
        d = get_samples(dataset, MockParser())
        aencoder = ActionSequenceEncoder(d, 0)
        input = TransformCode(MockParser())(Environment(
            supervisions={"ground_truth": "y = x + 1"}
        ))
        transform = TransformGroundTruth(aencoder)
        input.states["reference"] = [Token(None, "foo", "foo"),
                                     Token(None, "bar", "bar")]
        ground_truth = transform(input).supervisions["ground_truth_actions"]
        assert np.array_equal(
            [
                [3, -1, -1], [4, -1, -1], [-1, 1, -1], [1, -1, -1],
                [5, -1, -1], [-1, 2, -1], [1, -1, -1], [4, -1, -1],
                [-1, 3, -1], [1, -1, -1], [6, -1, -1], [-1, 4, -1],
                [1, -1, -1]
            ],
            ground_truth.numpy()
        )

    def test_impossible_case(self):
        entries = [Environment(
            inputs={"input": "foo bar"},
            supervisions={"ground_truth": "y = x + 1"}
        )]
        dataset = ListDataset(entries)
        d = get_samples(dataset, MockParser())
        d.tokens = [("", "y"), ("", "1")]
        aencoder = ActionSequenceEncoder(d, 0)
        action_sequence = TransformCode(MockParser())(Environment(
            supervisions={"ground_truth": "y = x + 1"}
        )).supervisions["action_sequence"]
        transform = TransformGroundTruth(aencoder)
        ground_truth = transform(Environment(
            states={"reference": [Token(None, "foo", "foo"),
                                  Token(None, "bar", "bar")]},
            supervisions={"action_sequence": action_sequence}
        ))
        assert ground_truth is None


class TestTransformActionSequenceForRnnDecoder(object):
    def test_simple_case(self):
        entries = [Environment(
            inputs={"input": "foo bar"},
            supervisions={"ground_truth": "y = x + 1"}
        )]
        dataset = ListDataset(entries)
        d = get_samples(dataset, MockParser())
        aencoder = ActionSequenceEncoder(d, 0)
        transform = TransformActionSequenceForRnnDecoder(aencoder)
        action_sequence = TransformCode(MockParser())(Environment(
            supervisions={"ground_truth": "y = x + 1"}
        )).supervisions["action_sequence"]
        result = transform(Environment(
            states={"reference": [Token(None, "foo", "foo"),
                                  Token(None, "bar", "bar")]},
            supervisions={"action_sequence": action_sequence}
        ))
        prev_action_tensor = result.states["previous_actions"]
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
        entries = [Environment(
            inputs={"input": "foo bar"},
            supervisions={"ground_truth": "y = x + 1"}
        )]
        dataset = ListDataset(entries)
        d = get_samples(dataset, MockParser())
        aencoder = ActionSequenceEncoder(d, 0)
        action_sequence = TransformCode(MockParser())(Environment(
            supervisions={"ground_truth": "y = x + 1"}
        )).supervisions["action_sequence"]
        transform = TransformActionSequenceForRnnDecoder(aencoder, train=False)
        result = transform(Environment(
            states={"reference": [Token(None, "foo", "foo"),
                                  Token(None, "bar", "bar")],
                    "action_sequence": action_sequence}
        ))
        prev_action_tensor = result.states["previous_actions"]

        assert np.array_equal(
            [[1, -1, -1]],
            prev_action_tensor.numpy()
        )

    def test_impossible_case(self):
        entries = [Environment(
            inputs={"input": "foo bar"},
            supervisions={"ground_truth": "y = x + 1"}
        )]
        dataset = ListDataset(entries)
        d = get_samples(dataset, MockParser())
        d.tokens = [("", "y"), ("", "1")]
        aencoder = ActionSequenceEncoder(d, 0)
        transform = TransformActionSequenceForRnnDecoder(aencoder)
        action_sequence = TransformCode(MockParser())(Environment(
            supervisions={"ground_truth": "y = x + 1"}
        )).supervisions["action_sequence"]
        result = transform(Environment(
            states={"reference": [Token(None, "foo", "foo"),
                                  Token(None, "bar", "bar")]},
            supervisions={"action_sequence": action_sequence}
        ))
        assert result is None
