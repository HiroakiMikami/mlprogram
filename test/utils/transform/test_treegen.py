import numpy as np
from typing import List
from torchnlp.encoders import LabelEncoder
from mlprogram.utils.data import ListDataset, get_samples
from mlprogram.languages import Node, Field, Leaf
from mlprogram.languages import Token
from mlprogram.languages import Parser
from mlprogram.encoders import ActionSequenceEncoder
from mlprogram.utils.transform.action_sequence import TransformCode
from mlprogram.utils.transform.treegen \
    import TransformQuery, TransformActionSequence


def tokenize(str: str) -> List[Token]:
    return list(map(lambda x: Token(None, x, x), str.split(" ")))


class MockParser(Parser[str]):
    def parse(self, code):
        ast = Node("Assign",
                   [Field("name", "Name",
                          Node("Name", [Field("id", "str",
                                              Leaf("str", "x"))])),
                    Field("value", "expr",
                          Node("Op", [
                              Field("op", "str", Leaf("str", "+")),
                              Field("arg0", "expr",
                                    Node("Name", [Field("id", "str",
                                                        Leaf("str", "y"))])),
                              Field("arg1", "expr",
                                    Node("Number", [
                                        Field("value", "number",
                                              Leaf("number", "1"))]))]
                               ))])
        return ast


class TestTransformQuery(object):
    def test_simple_case(self):
        words = ["ab", "test"]
        qencoder = LabelEncoder(words, 0)
        cencoder = LabelEncoder(["a", "b", "t", "e"], 0)
        transform = TransformQuery(tokenize, qencoder, cencoder, 3)
        result = transform({"input": "ab test"})
        reference = result["reference"]
        word_query = result["word_nl_query"]
        char_query = result["char_nl_query"]
        assert [Token(None, "ab", "ab"),
                Token(None, "test", "test")] == reference
        assert np.array_equal([1, 2], word_query.numpy())
        assert np.array_equal([[1, 2, -1], [3, 4, 0]],
                              char_query.numpy())


class TestTransformActionSequence(object):
    def test_simple_case(self):
        entries = [{"input": "ab test", "ground_truth": "y = x + 1"}]
        dataset = ListDataset(entries)
        d = get_samples(dataset, MockParser())
        aencoder = ActionSequenceEncoder(d, 0)
        action_sequence = \
            TransformCode(MockParser())({
                "ground_truth": "y = x + 1"
            })["action_sequence"]
        transform = TransformActionSequence(aencoder, 2, 3)
        result = transform({
            "action_sequence": action_sequence,
            "reference": [Token(None, "ab", "ab"), Token(None, "test", "test")]
        })
        prev_action = result["previous_actions"]
        prev_rule_action = result["previous_action_rules"]
        depth = result["depthes"]
        matrix = result["adjacency_matrix"]
        query = result["action_queries"]
        assert np.array_equal(
            [
                [2, -1, -1], [3, -1, -1], [4, -1, -1], [-1, 1, -1],
                [5, -1, -1], [-1, 2, -1], [4, -1, -1], [-1, 3, -1],
                [6, -1, -1]
            ],
            prev_action.numpy()
        )
        assert np.array_equal(
            [
                # None -> Root
                [[1, -1, -1], [2, -1, -1], [-1, -1, -1]],
                # Assign -> Name, expr
                [[3, -1, -1], [4, -1, -1], [5, -1, -1]],
                # Name -> str
                [[4, -1, -1], [6, -1, -1], [-1, -1, -1]],
                # str -> "x"
                [[-1, -1, -1], [-1, 1, -1], [-1, -1, -1]],
                # Op -> str, expr, expr
                [[7, -1, -1], [6, -1, -1], [5, -1, -1]],
                # str -> "+"
                [[-1, -1, -1], [-1, 2, -1], [-1, -1, -1]],
                # Name -> str
                [[4, -1, -1], [6, -1, -1], [-1, -1, -1]],
                # str -> "y"
                [[-1, -1, -1], [-1, 3, -1], [-1, -1, -1]],
                # Number -> number
                [[8, -1, -1], [9, -1, -1], [-1, -1, -1]],
            ],
            prev_rule_action.numpy()
        )
        assert np.array_equal(
            [0, 1, 2, 3, 2, 3, 3, 4, 3],
            depth.numpy()
        )
        assert np.array_equal(
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
        )
        assert np.array_equal(
            [
                [-1, -1, -1], [2, -1, -1], [3, 2, -1], [4, 3, 2],
                [3, 2, -1], [5, 3, 2], [5, 3, 2], [4, 5, 3],
                [5, 3, 2]
            ],
            query.numpy()
        )

    def test_eval(self):
        entries = [{"input": "ab test", "ground_truth": "y = x + 1"}]
        dataset = ListDataset(entries)
        d = get_samples(dataset, MockParser())
        aencoder = ActionSequenceEncoder(d, 0)
        action_sequence = \
            TransformCode(MockParser())({
                "ground_truth": "y = x + 1"
            })["action_sequence"]
        transform = TransformActionSequence(aencoder, 2, 3, train=False)
        result = transform({
            "action_sequence": action_sequence,
            "reference": [Token(None, "ab", "ab"), Token(None, "test", "test")]
        })
        prev_action = result["previous_actions"]
        prev_rule_action = result["previous_action_rules"]
        depth = result["depthes"]
        matrix = result["adjacency_matrix"]
        query = result["action_queries"]
        assert np.array_equal(
            [
                [2, -1, -1], [3, -1, -1], [4, -1, -1], [-1, 1, -1],
                [5, -1, -1], [-1, 2, -1], [4, -1, -1], [-1, 3, -1],
                [6, -1, -1], [-1, 4, -1]
            ],
            prev_action.numpy()
        )
        assert np.array_equal(
            [
                # None -> Root
                [[1, -1, -1], [2, -1, -1], [-1, -1, -1]],
                # Assign -> Name, expr
                [[3, -1, -1], [4, -1, -1], [5, -1, -1]],
                # Name -> str
                [[4, -1, -1], [6, -1, -1], [-1, -1, -1]],
                # str -> "x"
                [[-1, -1, -1], [-1, 1, -1], [-1, -1, -1]],
                # Op -> str, expr, expr
                [[7, -1, -1], [6, -1, -1], [5, -1, -1]],
                # str -> "+"
                [[-1, -1, -1], [-1, 2, -1], [-1, -1, -1]],
                # Name -> str
                [[4, -1, -1], [6, -1, -1], [-1, -1, -1]],
                # str -> "y"
                [[-1, -1, -1], [-1, 3, -1], [-1, -1, -1]],
                # Number -> number
                [[8, -1, -1], [9, -1, -1], [-1, -1, -1]],
                [[-1, -1, -1], [-1, 4, -1], [-1, -1, -1]],
            ],
            prev_rule_action.numpy()
        )
        assert np.array_equal(
            [0, 1, 2, 3, 2, 3, 3, 4, 3, 4],
            depth.numpy()
        )
        assert np.array_equal(
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
        )
        assert np.array_equal(
            [
                [-1, -1, -1], [2, -1, -1], [3, 2, -1], [4, 3, 2],
                [3, 2, -1], [5, 3, 2], [5, 3, 2], [4, 5, 3],
                [5, 3, 2], [6, 5, 3]
            ],
            query.numpy()
        )

    def test_impossible_case(self):
        entries = [{"input": "foo bar", "ground_truth": "y = x + 1"}]
        dataset = ListDataset(entries)
        d = get_samples(dataset, MockParser())
        d.tokens = [("", "y"), ("", "1")]
        aencoder = ActionSequenceEncoder(d, 0)
        action_sequence = \
            TransformCode(MockParser())({
                "ground_truth": "y = x + 1"
            })["action_sequence"]
        transform = TransformActionSequence(aencoder, 3, 3)
        result = transform({
            "action_sequence": action_sequence,
            "reference": [Token(None, "ab", "ab"), Token(None, "test", "test")]
        })
        assert result is None
