import numpy as np
import pytest

from mlprogram.builtins import Environment
from mlprogram.encoders import ActionSequenceEncoder
from mlprogram.languages import Field, Leaf, Node, Parser, Token
from mlprogram.transforms.action_sequence import (
    AddActions,
    AddActionSequenceAsTree,
    AddPreviousActionRules,
    AddPreviousActions,
    AddQueryForTreeGenDecoder,
    AddState,
    EncodeActionSequence,
    GroundTruthToActionSequence,
)
from mlprogram.utils.data import ListDataset, get_samples


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


class MockParserWithoutVariadicArgs(Parser[str]):
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


class TestGroundTruthToActionSequence(object):
    def test_simple_case(self):
        transform = GroundTruthToActionSequence(MockParser())
        action_sequence = transform(ground_truth="y = x + 1")
        assert action_sequence.head is None


class TestEncodeActionSequence(object):
    def test_simple_case(self):
        entries = [Environment(
            {"ground_truth": "y = x + 1"},
            set(["ground_truth"])
        )]
        dataset = ListDataset(entries)
        d = get_samples(dataset, MockParser())
        aencoder = ActionSequenceEncoder(d, 0)
        action_sequence = GroundTruthToActionSequence(MockParser())(
            ground_truth="y = x + 1"
        )
        transform = EncodeActionSequence(aencoder)
        ground_truth = transform(
            action_sequence=action_sequence,
            reference=[Token(None, "foo", "foo"), Token(None, "bar", "bar")],
        )
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
            {"ground_truth": "y = x + 1"},
            set(["ground_truth"])
        )]
        dataset = ListDataset(entries)
        d = get_samples(dataset, MockParser())
        d.tokens = [("", "y"), ("", "1")]
        aencoder = ActionSequenceEncoder(d, 0)
        action_sequence = GroundTruthToActionSequence(MockParser())(
            ground_truth="y = x + 1"
        )
        transform = EncodeActionSequence(aencoder)
        with pytest.raises(RuntimeError):
            transform(
                reference=[Token(None, "foo", "foo"), Token(None, "bar", "bar")],
                action_sequence=action_sequence,
            )


class TestAddPreviousActions(object):
    def test_simple_case(self):
        entries = [Environment(
            {"ground_truth": "y = x + 1"},
            set(["ground_truth"])
        )]
        dataset = ListDataset(entries)
        d = get_samples(dataset, MockParser())
        aencoder = ActionSequenceEncoder(d, 0)
        transform = AddPreviousActions(aencoder)
        action_sequence = GroundTruthToActionSequence(MockParser())(
            "y=x+1"
        )
        prev_action_tensor = transform(
            reference=[Token(None, "foo", "foo"), Token(None, "bar", "bar")],
            action_sequence=action_sequence,
            train=True
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
        entries = [Environment(
            {"ground_truth": "y = x + 1"},
            set(["ground_truth"])
        )]
        dataset = ListDataset(entries)
        d = get_samples(dataset, MockParser())
        aencoder = ActionSequenceEncoder(d, 0)
        action_sequence = GroundTruthToActionSequence(MockParser())(
            "y = x + 1"
        )
        transform = AddPreviousActions(aencoder)
        prev_action_tensor = transform(
            reference=[Token(None, "foo", "foo"), Token(None, "bar", "bar")],
            action_sequence=action_sequence,
            train=False
        )

        assert np.array_equal(
            [
                [2, -1, -1], [3, -1, -1], [4, -1, -1], [-1, 1, -1],
                [1, -1, -1], [5, -1, -1], [-1, 2, -1], [1, -1, -1],
                [4, -1, -1], [-1, 3, -1], [1, -1, -1], [6, -1, -1],
                [-1, 4, -1], [1, -1, -1]
            ],
            prev_action_tensor.numpy()
        )

    def test_n_dependent(self):
        entries = [Environment(
            {"ground_truth": "y = x + 1"},
            set(["ground_truth"])
        )]
        dataset = ListDataset(entries)
        d = get_samples(dataset, MockParser())
        aencoder = ActionSequenceEncoder(d, 0)
        action_sequence = GroundTruthToActionSequence(MockParser())(
            "y = x + 1"
        )
        transform = AddPreviousActions(aencoder, n_dependent=2)
        prev_action_tensor = transform(
            reference=[Token(None, "foo", "foo"), Token(None, "bar", "bar")],
            action_sequence=action_sequence,
            train=False
        )

        assert np.array_equal(
            [[-1, 4, -1], [1, -1, -1]],
            prev_action_tensor.numpy()
        )

    def test_impossible_case(self):
        entries = [Environment(
            {"ground_truth": "y = x + 1"},
            set(["ground_truth"])
        )]
        dataset = ListDataset(entries)
        d = get_samples(dataset, MockParser())
        d.tokens = [("", "y"), ("", "1")]
        aencoder = ActionSequenceEncoder(d, 0)
        transform = AddPreviousActions(aencoder)
        action_sequence = GroundTruthToActionSequence(MockParser())(
            "y = x + 1"
        )
        with pytest.raises(RuntimeError):
            transform(
                reference=[Token(None, "foo", "foo"), Token(None, "bar", "bar")],
                action_sequence=action_sequence,
                train=True
            )


class TestAddActions(object):
    def test_simple_case(self):
        entries = [Environment(
            {"text_query": "foo bar", "ground_truth": "y = x + 1"},
            set(["ground_truth"])
        )]
        dataset = ListDataset(entries)
        d = get_samples(dataset, MockParser())
        aencoder = ActionSequenceEncoder(d, 0)
        transform = AddActions(aencoder)
        action_sequence = GroundTruthToActionSequence(MockParser())(
            "y = x + 1"
        )
        action_tensor = transform(
            reference=[Token(None, "foo", "foo"), Token(None, "bar", "bar")],
            action_sequence=action_sequence,
            train=True
        )
        assert np.array_equal(
            [
                [2, 2, 0], [4, 3, 1], [6, 4, 2], [6, 4, 2], [5, 3, 1],
                [6, 5, 5], [6, 5, 5], [5, 5, 5], [6, 4, 8], [6, 4, 8],
                [5, 5, 5], [9, 6, 11], [9, 6, 11]
            ],
            action_tensor.numpy()
        )

    def test_eval(self):
        entries = [Environment(
            {"text_query": "foo bar", "ground_truth": "y = x + 1"},
            set(["ground_truth"])
        )]
        dataset = ListDataset(entries)
        d = get_samples(dataset, MockParser())
        aencoder = ActionSequenceEncoder(d, 0)
        action_sequence = GroundTruthToActionSequence(MockParser())(
            "y = x + 1"
        )
        transform = AddActions(aencoder)
        action_tensor = transform(
            reference=[Token(None, "foo", "foo"), Token(None, "bar", "bar")],
            action_sequence=action_sequence,
            train=False
        )

        assert np.array_equal(
            [
                [2, 2, 0], [4, 3, 1], [6, 4, 2], [6, 4, 2], [5, 3, 1],
                [6, 5, 5], [6, 5, 5], [5, 5, 5], [6, 4, 8], [6, 4, 8],
                [5, 5, 5], [9, 6, 11], [9, 6, 11], [-1, -1, -1]
            ],
            action_tensor.numpy()
        )

    def test_n_dependent(self):
        entries = [Environment(
            {"text_query": "foo bar", "ground_truth": "y = x + 1"},
            set(["ground_truth"])
        )]
        dataset = ListDataset(entries)
        d = get_samples(dataset, MockParser())
        aencoder = ActionSequenceEncoder(d, 0)
        action_sequence = GroundTruthToActionSequence(MockParser())(
            "y = x + 1"
        )
        transform = AddActions(aencoder, n_dependent=2)
        action_tensor = transform(
            reference=[Token(None, "foo", "foo"), Token(None, "bar", "bar")],
            action_sequence=action_sequence,
            train=False
        )

        assert np.array_equal(
            [[9, 6, 11], [-1, -1, -1]],
            action_tensor.numpy()
        )

    def test_impossible_case(self):
        entries = [Environment(
            {"text_query": "foo bar", "ground_truth": "y = x + 1"},
            set(["ground_truth"])
        )]
        dataset = ListDataset(entries)
        d = get_samples(dataset, MockParser())
        d.tokens = [("", "y"), ("", "1")]
        aencoder = ActionSequenceEncoder(d, 0)
        transform = AddActions(aencoder)
        action_sequence = GroundTruthToActionSequence(MockParser())(
            "y = x + 1"
        )
        with pytest.raises(RuntimeError):
            transform(
                reference=[Token(None, "foo", "foo"), Token(None, "bar", "bar")],
                action_sequence=action_sequence,
                train=True
            )


class TestAddPreviousActionRules(object):
    def test_simple_case(self):
        entries = [Environment(
            {"text_query": "ab test", "ground_truth": "y = x + 1"},
            set(["ground_truth"])
        )]
        dataset = ListDataset(entries)
        d = get_samples(dataset, MockParserWithoutVariadicArgs())
        aencoder = ActionSequenceEncoder(d, 0)
        action_sequence = GroundTruthToActionSequence(
            MockParserWithoutVariadicArgs())(
                "y = x + 1"
        )
        transform = AddPreviousActionRules(aencoder, 2)
        prev_rule_action = transform(
            reference=[Token(None, "ab", "ab"), Token(None, "test", "test")],
            action_sequence=action_sequence,
            train=True
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

    def test_eval(self):
        entries = [Environment(
            {"text_query": "ab test", "ground_truth": "y = x + 1"},
            set(["ground_truth"])
        )]
        dataset = ListDataset(entries)
        d = get_samples(dataset, MockParserWithoutVariadicArgs())
        aencoder = ActionSequenceEncoder(d, 0)
        action_sequence = GroundTruthToActionSequence(MockParserWithoutVariadicArgs())(
            "y = x + 1"
        )
        transform = AddPreviousActionRules(aencoder, 2)
        prev_rule_action = transform(
            reference=[Token(None, "ab", "ab"), Token(None, "test", "test")],
            action_sequence=action_sequence,
            train=False
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

    def test_n_dependent(self):
        entries = [Environment(
            {"text_query": "ab test", "ground_truth": "y = x + 1"},
            set(["ground_truth"])
        )]
        dataset = ListDataset(entries)
        d = get_samples(dataset, MockParserWithoutVariadicArgs())
        aencoder = ActionSequenceEncoder(d, 0)
        action_sequence = GroundTruthToActionSequence(MockParserWithoutVariadicArgs())(
            "y = x + 1"
        )
        transform = AddPreviousActionRules(aencoder, 2, n_dependent=3)
        prev_rule_action = transform(
            reference=[Token(None, "ab", "ab"), Token(None, "test", "test")],
            action_sequence=action_sequence,
            train=False,
        )
        assert np.array_equal(
            [
                # str -> "y"
                [[-1, -1, -1], [-1, 3, -1], [-1, -1, -1]],
                # Number -> number
                [[8, -1, -1], [9, -1, -1], [-1, -1, -1]],
                [[-1, -1, -1], [-1, 4, -1], [-1, -1, -1]],
            ],
            prev_rule_action.numpy()
        )


class TestAddActionSequenceAsTree(object):
    def test_simple_case(self):
        entries = [Environment(
            {"text_query": "ab test", "ground_truth": "y = x + 1"},
            set(["ground_truth"])
        )]
        dataset = ListDataset(entries)
        d = get_samples(dataset, MockParserWithoutVariadicArgs())
        aencoder = ActionSequenceEncoder(d, 0)
        action_sequence = GroundTruthToActionSequence(MockParserWithoutVariadicArgs())(
            "y = x + 1"
        )
        transform = AddActionSequenceAsTree(aencoder)
        matrix, depth = transform(
            reference=[Token(None, "ab", "ab"), Token(None, "test", "test")],
            action_sequence=action_sequence,
            train=True
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

    def test_eval(self):
        entries = [Environment(
            {"text_query": "ab test", "ground_truth": "y = x + 1"},
            set(["ground_truth"])
        )]
        dataset = ListDataset(entries)
        d = get_samples(dataset, MockParserWithoutVariadicArgs())
        aencoder = ActionSequenceEncoder(d, 0)
        action_sequence = GroundTruthToActionSequence(MockParserWithoutVariadicArgs())(
            "y = x + 1"
        )
        transform = AddActionSequenceAsTree(aencoder,)
        matrix, depth = transform(
            reference=[Token(None, "ab", "ab"), Token(None, "test", "test")],
            action_sequence=action_sequence,
            train=False
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


class TestAddQueryForTreeGenDecoder(object):
    def test_simple_case(self):
        entries = [Environment(
            {"text_query": "ab test", "ground_truth": "y = x + 1"},
            set(["ground_truth"])
        )]
        dataset = ListDataset(entries)
        d = get_samples(dataset, MockParserWithoutVariadicArgs())
        aencoder = ActionSequenceEncoder(d, 0)
        action_sequence = GroundTruthToActionSequence(MockParserWithoutVariadicArgs())(
            "y = x + 1"
        )
        transform = AddQueryForTreeGenDecoder(aencoder, 3)
        query = transform(
            reference=[Token(None, "ab", "ab"), Token(None, "test", "test")],
            action_sequence=action_sequence,
            train=True
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
        entries = [Environment(
            {"text_query": "ab test", "ground_truth": "y = x + 1"},
            set(["ground_truth"])
        )]
        dataset = ListDataset(entries)
        d = get_samples(dataset, MockParserWithoutVariadicArgs())
        aencoder = ActionSequenceEncoder(d, 0)
        action_sequence = GroundTruthToActionSequence(MockParserWithoutVariadicArgs())(
            "y = x + 1"
        )
        transform = AddQueryForTreeGenDecoder(aencoder, 3,)
        query = transform(
            reference=[Token(None, "ab", "ab"), Token(None, "test", "test")],
            action_sequence=action_sequence,
            train=False
        )
        assert np.array_equal(
            [
                [-1, -1, -1], [2, -1, -1], [3, 2, -1], [4, 3, 2],
                [3, 2, -1], [5, 3, 2], [5, 3, 2], [4, 5, 3],
                [5, 3, 2], [6, 5, 3]
            ],
            query.numpy()
        )

    def test_n_dependent(self):
        entries = [Environment(
            {"text_query": "ab test", "ground_truth": "y = x + 1"},
            set(["ground_truth"])
        )]
        dataset = ListDataset(entries)
        d = get_samples(dataset, MockParserWithoutVariadicArgs())
        aencoder = ActionSequenceEncoder(d, 0)
        action_sequence = GroundTruthToActionSequence(MockParserWithoutVariadicArgs())(
            "y = x + 1"
        )
        transform = AddQueryForTreeGenDecoder(aencoder, 3, n_dependent=2)
        query = transform(
            reference=[Token(None, "ab", "ab"), Token(None, "test", "test")],
            action_sequence=action_sequence,
            train=False
        )
        assert np.array_equal(
            [[5, 3, 2], [6, 5, 3]],
            query.numpy()
        )


class TestAddState(object):
    def test_simple_case(self):
        transform = AddState("key", None)
        result = transform(Environment({"train": True}))
        assert result["key"] is None

    def test_eval(self):
        transform = AddState("key", None)
        result = transform(Environment({"train": False}))
        assert result["key"] is None

    def test_eval2(self):
        transform = AddState("key", None)
        result = transform(Environment({"train": False, "key": 2}))
        assert result["key"] == 2
