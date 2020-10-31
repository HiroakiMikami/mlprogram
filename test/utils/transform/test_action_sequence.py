import numpy as np

from mlprogram import Environment
from mlprogram.utils.data import ListDataset, get_samples
from mlprogram.languages import Node, Leaf, Field
from mlprogram.languages import Token
from mlprogram.languages import Parser
from mlprogram.encoders import ActionSequenceEncoder
from mlprogram.utils.transform.action_sequence \
    import GroundTruthToActionSequence, EncodeActionSequence
from mlprogram.utils.transform.action_sequence \
    import AddPreviousActions
from mlprogram.utils.transform.action_sequence import AddActions
from mlprogram.utils.transform.action_sequence import AddPreviousActionRules
from mlprogram.utils.transform.action_sequence import AddActionSequenceAsTree
from mlprogram.utils.transform.action_sequence \
    import AddStateForRnnDecoder
from mlprogram.utils.transform.action_sequence import AddHistoryState
from mlprogram.utils.transform.action_sequence import AddQueryForTreeGenDecoder


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
        action_sequence = transform(Environment(
            supervisions={"ground_truth": "y = x + 1"}
        )).supervisions["action_sequence"]
        assert action_sequence.head is None


class TestEncodeActionSequence(object):
    def test_simple_case(self):
        entries = [Environment(
            supervisions={"ground_truth": "y = x + 1"}
        )]
        dataset = ListDataset(entries)
        d = get_samples(dataset, MockParser())
        aencoder = ActionSequenceEncoder(d, 0)
        input = GroundTruthToActionSequence(MockParser())(Environment(
            supervisions={"ground_truth": "y = x + 1"}
        ))
        transform = EncodeActionSequence(aencoder)
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
            supervisions={"ground_truth": "y = x + 1"}
        )]
        dataset = ListDataset(entries)
        d = get_samples(dataset, MockParser())
        d.tokens = [("", "y"), ("", "1")]
        aencoder = ActionSequenceEncoder(d, 0)
        action_sequence = GroundTruthToActionSequence(MockParser())(
            Environment(supervisions={"ground_truth": "y = x + 1"})
        ).supervisions["action_sequence"]
        transform = EncodeActionSequence(aencoder)
        ground_truth = transform(Environment(
            states={"reference": [Token(None, "foo", "foo"),
                                  Token(None, "bar", "bar")]},
            supervisions={"action_sequence": action_sequence}
        ))
        assert ground_truth is None


class TestAddPreviousActions(object):
    def test_simple_case(self):
        entries = [Environment(
            supervisions={"ground_truth": "y = x + 1"}
        )]
        dataset = ListDataset(entries)
        d = get_samples(dataset, MockParser())
        aencoder = ActionSequenceEncoder(d, 0)
        transform = AddPreviousActions(aencoder)
        action_sequence = GroundTruthToActionSequence(MockParser())(
            Environment(supervisions={"ground_truth": "y = x + 1"})
        ).supervisions["action_sequence"]
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
            supervisions={"ground_truth": "y = x + 1"}
        )]
        dataset = ListDataset(entries)
        d = get_samples(dataset, MockParser())
        aencoder = ActionSequenceEncoder(d, 0)
        action_sequence = GroundTruthToActionSequence(MockParser())(
            Environment(supervisions={"ground_truth": "y = x + 1"})
        ).supervisions["action_sequence"]
        transform = AddPreviousActions(aencoder)
        result = transform(Environment(
            states={"reference": [Token(None, "foo", "foo"),
                                  Token(None, "bar", "bar")],
                    "action_sequence": action_sequence}
        ))
        prev_action_tensor = result.states["previous_actions"]

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
            supervisions={"ground_truth": "y = x + 1"}
        )]
        dataset = ListDataset(entries)
        d = get_samples(dataset, MockParser())
        aencoder = ActionSequenceEncoder(d, 0)
        action_sequence = GroundTruthToActionSequence(MockParser())(
            Environment(supervisions={"ground_truth": "y = x + 1"})
        ).supervisions["action_sequence"]
        transform = AddPreviousActions(aencoder, n_dependent=2)
        result = transform(Environment(
            states={"reference": [Token(None, "foo", "foo"),
                                  Token(None, "bar", "bar")],
                    "action_sequence": action_sequence}
        ))
        prev_action_tensor = result.states["previous_actions"]

        assert np.array_equal(
            [[-1, 4, -1], [1, -1, -1]],
            prev_action_tensor.numpy()
        )

    def test_impossible_case(self):
        entries = [Environment(
            supervisions={"ground_truth": "y = x + 1"}
        )]
        dataset = ListDataset(entries)
        d = get_samples(dataset, MockParser())
        d.tokens = [("", "y"), ("", "1")]
        aencoder = ActionSequenceEncoder(d, 0)
        transform = AddPreviousActions(aencoder)
        action_sequence = GroundTruthToActionSequence(MockParser())(
            Environment(supervisions={"ground_truth": "y = x + 1"})
        ).supervisions["action_sequence"]
        result = transform(Environment(
            states={"reference": [Token(None, "foo", "foo"),
                                  Token(None, "bar", "bar")]},
            supervisions={"action_sequence": action_sequence}
        ))
        assert result is None


class TestAddActions(object):
    def test_simple_case(self):
        entries = [Environment(
            inputs={"text_query": "foo bar"},
            supervisions={"ground_truth": "y = x + 1"}
        )]
        dataset = ListDataset(entries)
        d = get_samples(dataset, MockParser())
        aencoder = ActionSequenceEncoder(d, 0)
        transform = AddActions(aencoder)
        action_sequence = GroundTruthToActionSequence(MockParser())(
            Environment(supervisions={"ground_truth": "y = x + 1"})
        ).supervisions["action_sequence"]
        result = transform(Environment(
            states={"reference": [Token(None, "foo", "foo"),
                                  Token(None, "bar", "bar")]},
            supervisions={"action_sequence": action_sequence}
        ))
        action_tensor = result.states["actions"]
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
            inputs={"text_query": "foo bar"},
            supervisions={"ground_truth": "y = x + 1"}
        )]
        dataset = ListDataset(entries)
        d = get_samples(dataset, MockParser())
        aencoder = ActionSequenceEncoder(d, 0)
        action_sequence = GroundTruthToActionSequence(MockParser())(
            Environment(supervisions={"ground_truth": "y = x + 1"})
        ).supervisions["action_sequence"]
        transform = AddActions(aencoder)
        result = transform(Environment(
            states={"reference": [Token(None, "foo", "foo"),
                                  Token(None, "bar", "bar")],
                    "action_sequence": action_sequence}
        ))
        action_tensor = result.states["actions"]

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
            inputs={"text_query": "foo bar"},
            supervisions={"ground_truth": "y = x + 1"}
        )]
        dataset = ListDataset(entries)
        d = get_samples(dataset, MockParser())
        aencoder = ActionSequenceEncoder(d, 0)
        action_sequence = GroundTruthToActionSequence(MockParser())(
            Environment(supervisions={"ground_truth": "y = x + 1"})
        ).supervisions["action_sequence"]
        transform = AddActions(aencoder, n_dependent=2)
        result = transform(Environment(
            states={"reference": [Token(None, "foo", "foo"),
                                  Token(None, "bar", "bar")],
                    "action_sequence": action_sequence}
        ))
        action_tensor = result.states["actions"]

        assert np.array_equal(
            [[9, 6, 11], [-1, -1, -1]],
            action_tensor.numpy()
        )

    def test_impossible_case(self):
        entries = [Environment(
            inputs={"text_query": "foo bar"},
            supervisions={"ground_truth": "y = x + 1"}
        )]
        dataset = ListDataset(entries)
        d = get_samples(dataset, MockParser())
        d.tokens = [("", "y"), ("", "1")]
        aencoder = ActionSequenceEncoder(d, 0)
        transform = AddActions(aencoder)
        action_sequence = GroundTruthToActionSequence(MockParser())(
            Environment(supervisions={"ground_truth": "y = x + 1"})
        ).supervisions["action_sequence"]
        result = transform(Environment(
            states={"reference": [Token(None, "foo", "foo"),
                                  Token(None, "bar", "bar")]},
            supervisions={"action_sequence": action_sequence}
        ))
        assert result is None


class TestAddPreviousActionRules(object):
    def test_simple_case(self):
        entries = [Environment(
            inputs={"text_query": "ab test"},
            supervisions={"ground_truth": "y = x + 1"}
        )]
        dataset = ListDataset(entries)
        d = get_samples(dataset, MockParserWithoutVariadicArgs())
        aencoder = ActionSequenceEncoder(d, 0)
        action_sequence = GroundTruthToActionSequence(
            MockParserWithoutVariadicArgs())(
                Environment(supervisions={"ground_truth": "y = x + 1"})
        ).supervisions["action_sequence"]
        transform = AddPreviousActionRules(aencoder, 2)
        result = transform(Environment(
            states={"reference": [Token(None, "ab", "ab"),
                                  Token(None, "test", "test")]},
            supervisions={"action_sequence": action_sequence}
        ))
        prev_rule_action = result.states["previous_action_rules"]
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
            inputs={"text_query": "ab test"},
            supervisions={"ground_truth": "y = x + 1"}
        )]
        dataset = ListDataset(entries)
        d = get_samples(dataset, MockParserWithoutVariadicArgs())
        aencoder = ActionSequenceEncoder(d, 0)
        action_sequence = GroundTruthToActionSequence(
            MockParserWithoutVariadicArgs())(
                Environment(supervisions={"ground_truth": "y = x + 1"})
        ).supervisions["action_sequence"]
        transform = AddPreviousActionRules(aencoder, 2)
        result = transform(Environment(
            states={"reference": [Token(None, "ab", "ab"),
                                  Token(None, "test", "test")],
                    "action_sequence": action_sequence}
        ))
        prev_rule_action = result.states["previous_action_rules"]
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
            inputs={"text_query": "ab test"},
            supervisions={"ground_truth": "y = x + 1"}
        )]
        dataset = ListDataset(entries)
        d = get_samples(dataset, MockParserWithoutVariadicArgs())
        aencoder = ActionSequenceEncoder(d, 0)
        action_sequence = GroundTruthToActionSequence(
            MockParserWithoutVariadicArgs())(
                Environment(supervisions={"ground_truth": "y = x + 1"})
        ).supervisions["action_sequence"]
        transform = AddPreviousActionRules(aencoder, 2, n_dependent=3)
        result = transform(Environment(
            states={"reference": [Token(None, "ab", "ab"),
                                  Token(None, "test", "test")],
                    "action_sequence": action_sequence}
        ))
        prev_rule_action = result.states["previous_action_rules"]
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
            inputs={"text_query": "ab test"},
            supervisions={"ground_truth": "y = x + 1"}
        )]
        dataset = ListDataset(entries)
        d = get_samples(dataset, MockParserWithoutVariadicArgs())
        aencoder = ActionSequenceEncoder(d, 0)
        action_sequence = GroundTruthToActionSequence(
            MockParserWithoutVariadicArgs())(
                Environment(supervisions={"ground_truth": "y = x + 1"})
        ).supervisions["action_sequence"]
        transform = AddActionSequenceAsTree(aencoder)
        result = transform(Environment(
            states={"reference": [Token(None, "ab", "ab"),
                                  Token(None, "test", "test")]},
            supervisions={"action_sequence": action_sequence}
        ))
        depth = result.states["depthes"]
        matrix = result.states["adjacency_matrix"]
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
            inputs={"text_query": "ab test"},
            supervisions={"ground_truth": "y = x + 1"}
        )]
        dataset = ListDataset(entries)
        d = get_samples(dataset, MockParserWithoutVariadicArgs())
        aencoder = ActionSequenceEncoder(d, 0)
        action_sequence = GroundTruthToActionSequence(
            MockParserWithoutVariadicArgs())(
                Environment(supervisions={"ground_truth": "y = x + 1"})
        ).supervisions["action_sequence"]
        transform = AddActionSequenceAsTree(aencoder,)
        result = transform(Environment(
            states={"reference": [Token(None, "ab", "ab"),
                                  Token(None, "test", "test")],
                    "action_sequence": action_sequence}
        ))
        depth = result.states["depthes"]
        matrix = result.states["adjacency_matrix"]
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
            inputs={"text_query": "ab test"},
            supervisions={"ground_truth": "y = x + 1"}
        )]
        dataset = ListDataset(entries)
        d = get_samples(dataset, MockParserWithoutVariadicArgs())
        aencoder = ActionSequenceEncoder(d, 0)
        action_sequence = GroundTruthToActionSequence(
            MockParserWithoutVariadicArgs())(
                Environment(supervisions={"ground_truth": "y = x + 1"})
        ).supervisions["action_sequence"]
        transform = AddQueryForTreeGenDecoder(aencoder, 3)
        result = transform(Environment(
            states={"reference": [Token(None, "ab", "ab"),
                                  Token(None, "test", "test")]},
            supervisions={"action_sequence": action_sequence}
        ))
        query = result.states["action_queries"]
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
            inputs={"text_query": "ab test"},
            supervisions={"ground_truth": "y = x + 1"}
        )]
        dataset = ListDataset(entries)
        d = get_samples(dataset, MockParserWithoutVariadicArgs())
        aencoder = ActionSequenceEncoder(d, 0)
        action_sequence = GroundTruthToActionSequence(
            MockParserWithoutVariadicArgs())(
                Environment(supervisions={"ground_truth": "y = x + 1"})
        ).supervisions["action_sequence"]
        transform = AddQueryForTreeGenDecoder(aencoder, 3,)
        result = transform(Environment(
            states={"reference": [Token(None, "ab", "ab"),
                                  Token(None, "test", "test")],
                    "action_sequence": action_sequence}
        ))
        query = result.states["action_queries"]
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
            inputs={"text_query": "ab test"},
            supervisions={"ground_truth": "y = x + 1"}
        )]
        dataset = ListDataset(entries)
        d = get_samples(dataset, MockParserWithoutVariadicArgs())
        aencoder = ActionSequenceEncoder(d, 0)
        action_sequence = GroundTruthToActionSequence(
            MockParserWithoutVariadicArgs())(
                Environment(supervisions={"ground_truth": "y = x + 1"})
        ).supervisions["action_sequence"]
        transform = AddQueryForTreeGenDecoder(aencoder, 3, n_dependent=2)
        result = transform(Environment(
            states={"reference": [Token(None, "ab", "ab"),
                                  Token(None, "test", "test")],
                    "action_sequence": action_sequence}
        ))
        query = result.states["action_queries"]
        assert np.array_equal(
            [[5, 3, 2], [6, 5, 3]],
            query.numpy()
        )


class TestAddStateForRnnDecoder(object):
    def test_simple_case(self):
        transform = AddStateForRnnDecoder()
        action_sequence = GroundTruthToActionSequence(MockParser())(
            Environment(supervisions={"ground_truth": "y = x + 1"})
        ).supervisions["action_sequence"]
        result = transform(Environment(
            states={"reference": [Token(None, "foo", "foo"),
                                  Token(None, "bar", "bar")]},
            supervisions={"action_sequence": action_sequence}
        ))
        assert result.states["state"] is None
        assert result.states["hidden_state"] is None

    def test_eval(self):
        action_sequence = GroundTruthToActionSequence(MockParser())(
            Environment(supervisions={"ground_truth": "y = x + 1"})
        ).supervisions["action_sequence"]
        transform = AddStateForRnnDecoder()
        result = transform(Environment(
            states={"reference": [Token(None, "foo", "foo"),
                                  Token(None, "bar", "bar")],
                    "action_sequence": action_sequence}
        ))

        assert result.states["state"] is None
        assert result.states["hidden_state"] is None


class TestAddHistoryState(object):
    def test_simple_case(self):
        transform = AddHistoryState()
        action_sequence = GroundTruthToActionSequence(MockParser())(
            Environment(supervisions={"ground_truth": "y = x + 1"})
        ).supervisions["action_sequence"]
        result = transform(Environment(
            states={"reference": [Token(None, "foo", "foo"),
                                  Token(None, "bar", "bar")]},
            supervisions={"action_sequence": action_sequence}
        ))
        assert result["state@history"] is None
