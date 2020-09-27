import torch
import unittest
import numpy as np

from mlprogram.actions \
    import ExpandTreeRule, NodeType, NodeConstraint, \
    ApplyRule, GenerateToken, CloseVariadicFieldRule
from mlprogram.actions import ActionSequence
from mlprogram.languages import Token
from mlprogram.encoders import Samples, ActionSequenceEncoder


class TestEncoder(unittest.TestCase):
    def test_reserved_labels(self):
        encoder = ActionSequenceEncoder(Samples([], [], []), 0)
        self.assertEqual(2, len(encoder._rule_encoder.vocab))
        self.assertEqual(1, len(encoder._token_encoder.vocab))

    def test_encode_raw_value(self):
        encoder = ActionSequenceEncoder(
            Samples([], [],
                    [Token("", "foo", "foo"), Token("x", "foo", "foo")]),
            0)
        self.assertEqual([1, 2], encoder.encode_raw_value("foo"))
        self.assertEqual([0], encoder.encode_raw_value("bar"))

    def test_encode_action(self):
        funcdef = ExpandTreeRule(
            NodeType("def", NodeConstraint.Node, False),
            [("name",
              NodeType("value", NodeConstraint.Token, True)),
             ("body",
              NodeType("expr", NodeConstraint.Node, True))])
        expr = ExpandTreeRule(
            NodeType("expr", NodeConstraint.Node, False),
            [("op", NodeType("value", NodeConstraint.Token, True)),
             ("arg0",
              NodeType("value", NodeConstraint.Token, True)),
             ("arg1",
              NodeType("value", NodeConstraint.Token, True))])

        encoder = ActionSequenceEncoder(
            Samples([funcdef, expr],
                    [NodeType("def", NodeConstraint.Node, False),
                     NodeType("value", NodeConstraint.Token, True),
                     NodeType("expr", NodeConstraint.Node, True)],
                    [Token("", "f", "f"), Token("", "2", "2")]),
            0)
        action_sequence = ActionSequence()
        action_sequence.eval(ApplyRule(funcdef))
        action_sequence.eval(GenerateToken(Token("", "f", "f")))
        action_sequence.eval(GenerateToken(Token("", "1", "1")))
        action_sequence.eval(GenerateToken(Token("", "2", "2")))
        action_sequence.eval(ApplyRule(CloseVariadicFieldRule()))
        action = encoder.encode_action(action_sequence,
                                       [Token("", "1", "1"),
                                        Token("", "2", "2")])

        self.assertTrue(np.array_equal(
            [
                [-1, 2, -1, -1],
                [2, -1, 1, -1],
                [2, -1, -1, 0],
                [2, -1, 2, 1],
                [2, 1, -1, -1],
                [3, -1, -1, -1]
            ],
            action.numpy()
        ))

    def test_encode_parent(self):
        funcdef = ExpandTreeRule(
            NodeType("def", NodeConstraint.Node, False),
            [("name",
              NodeType("value", NodeConstraint.Token, True)),
             ("body",
              NodeType("expr", NodeConstraint.Node, True))])
        expr = ExpandTreeRule(
            NodeType("expr", NodeConstraint.Node, False),
            [("op", NodeType("value", NodeConstraint.Token, True)),
             ("arg0",
              NodeType("value", NodeConstraint.Token, True)),
             ("arg1",
              NodeType("value", NodeConstraint.Token, True))])

        encoder = ActionSequenceEncoder(
            Samples([funcdef, expr],
                    [NodeType("def", NodeConstraint.Node, False),
                     NodeType("value", NodeConstraint.Token, True),
                     NodeType("expr", NodeConstraint.Node, False)],
                    [Token("", "f", "f"), Token("", "2", "2")]),
            0)
        action_sequence = ActionSequence()
        action_sequence.eval(ApplyRule(funcdef))
        action_sequence.eval(GenerateToken(Token("", "f", "f")))
        action_sequence.eval(GenerateToken(Token("", "1", "1")))
        action_sequence.eval(GenerateToken(Token("", "2", "2")))
        action_sequence.eval(ApplyRule(CloseVariadicFieldRule()))
        parent = encoder.encode_parent(action_sequence)

        self.assertTrue(np.array_equal(
            [
                [-1, -1, -1, -1],
                [1, 2, 0, 0],
                [1, 2, 0, 0],
                [1, 2, 0, 0],
                [1, 2, 0, 0],
                [1, 2, 0, 1]
            ],
            parent.numpy()
        ))

    def test_encode_tree(self):
        funcdef = ExpandTreeRule(
            NodeType("def", NodeConstraint.Node, False),
            [("name",
              NodeType("value", NodeConstraint.Token, True)),
             ("body",
              NodeType("expr", NodeConstraint.Node, True))])
        expr = ExpandTreeRule(
            NodeType("expr", NodeConstraint.Node, False),
            [("op", NodeType("value", NodeConstraint.Token, True)),
             ("arg0",
              NodeType("value", NodeConstraint.Token, True)),
             ("arg1",
              NodeType("value", NodeConstraint.Token, True))])

        encoder = ActionSequenceEncoder(
            Samples([funcdef, expr],
                    [NodeType("def", NodeConstraint.Node, False),
                     NodeType("value", NodeConstraint.Token, True),
                     NodeType("expr", NodeConstraint.Node, False)],
                    [Token("", "f", "f"), Token("", "2", "2")]),
            0)
        action_sequence = ActionSequence()
        action_sequence.eval(ApplyRule(funcdef))
        action_sequence.eval(GenerateToken(Token("", "f", "f")))
        action_sequence.eval(GenerateToken(Token("", "1", "1")))
        d, m = encoder.encode_tree(action_sequence)

        self.assertTrue(np.array_equal(
            [0, 1, 1], d.numpy()
        ))
        self.assertTrue(np.array_equal(
            [[0, 1, 1], [0, 0, 0], [0, 0, 0]],
            m.numpy()
        ))

    def test_encode_empty_sequence(self):
        funcdef = ExpandTreeRule(
            NodeType("def", NodeConstraint.Node, False),
            [("name",
              NodeType("value", NodeConstraint.Token, False)),
             ("body",
              NodeType("expr", NodeConstraint.Node, True))])
        expr = ExpandTreeRule(
            NodeType("expr", NodeConstraint.Node, False),
            [("op", NodeType("value", NodeConstraint.Token, False)),
             ("arg0",
              NodeType("value", NodeConstraint.Token, False)),
             ("arg1",
              NodeType("value", NodeConstraint.Token, False))])

        encoder = ActionSequenceEncoder(
            Samples([funcdef, expr],
                    [NodeType("def", NodeConstraint.Node, False),
                     NodeType("value", NodeConstraint.Token, False),
                     NodeType("expr", NodeConstraint.Node, False)],
                    [Token("", "f", "f")]),
            0)
        action_sequence = ActionSequence()
        action = encoder.encode_action(action_sequence, [Token("", "1", "1")])
        parent = encoder.encode_parent(action_sequence)
        d, m = encoder.encode_tree(action_sequence)

        self.assertTrue(np.array_equal(
            [
                [-1, -1, -1, -1]
            ],
            action.numpy()
        ))
        self.assertTrue(np.array_equal(
            [
                [-1, -1, -1, -1]
            ],
            parent.numpy()
        ))
        self.assertTrue(np.array_equal(np.zeros((0,)), d.numpy()))
        self.assertTrue(np.array_equal(np.zeros((0, 0)), m.numpy()))

    def test_encode_invalid_sequence(self):
        funcdef = ExpandTreeRule(
            NodeType("def", NodeConstraint.Node, False),
            [("name",
              NodeType("value", NodeConstraint.Token, True)),
             ("body",
              NodeType("expr", NodeConstraint.Node, True))])
        expr = ExpandTreeRule(
            NodeType("expr", NodeConstraint.Node, False),
            [("op", NodeType("value", NodeConstraint.Token, False)),
             ("arg0",
              NodeType("value", NodeConstraint.Token, True)),
             ("arg1",
              NodeType("value", NodeConstraint.Token, True))])

        encoder = ActionSequenceEncoder(
            Samples([funcdef, expr],
                    [NodeType("def", NodeConstraint.Node, False),
                     NodeType("value", NodeConstraint.Token, True),
                     NodeType("expr", NodeConstraint.Node, True)],
                    [Token("", "f", "f")]),
            0)
        action_sequence = ActionSequence()
        action_sequence.eval(ApplyRule(funcdef))
        action_sequence.eval(GenerateToken(Token("", "f", "f")))
        action_sequence.eval(GenerateToken(Token("", "1", "1")))
        action_sequence.eval(ApplyRule(CloseVariadicFieldRule()))

        self.assertEqual(None, encoder.encode_action(action_sequence,
                                                     [Token("", "2", "2")]))

    def test_encode_completed_sequence(self):
        none = ExpandTreeRule(NodeType("value", NodeConstraint.Node, False),
                              [])
        encoder = ActionSequenceEncoder(
            Samples([none],
                    [NodeType("value", NodeConstraint.Node, False)],
                    [Token("", "f", "f")]),
            0)
        action_sequence = ActionSequence()
        action_sequence.eval(ApplyRule(none))
        action = encoder.encode_action(action_sequence, [Token("", "1", "1")])
        parent = encoder.encode_parent(action_sequence)

        self.assertTrue(np.array_equal(
            [
                [-1, 2, -1, -1],
                [-1, -1, -1, -1]
            ],
            action.numpy()
        ))
        self.assertTrue(np.array_equal(
            [
                [-1, -1, -1, -1],
                [-1, -1, -1, -1]
            ],
            parent.numpy()
        ))

    def test_decode(self):
        funcdef = ExpandTreeRule(
            NodeType("def", NodeConstraint.Node, False),
            [("name",
              NodeType("value", NodeConstraint.Token, True)),
             ("body",
              NodeType("expr", NodeConstraint.Node, True))])
        expr = ExpandTreeRule(
            NodeType("expr", NodeConstraint.Node, False),
            [("op", NodeType("value", NodeConstraint.Token, True)),
             ("arg0",
              NodeType("value", NodeConstraint.Token, True)),
             ("arg1",
              NodeType("value", NodeConstraint.Token, True))])

        encoder = ActionSequenceEncoder(
            Samples([funcdef, expr],
                    [NodeType("def", NodeConstraint.Node, False),
                     NodeType("value", NodeConstraint.Token, True),
                     NodeType("expr", NodeConstraint.Node, False)],
                    [Token("", "f", "f")]),
            0)
        action_sequence = ActionSequence()
        action_sequence.eval(ApplyRule(funcdef))
        action_sequence.eval(GenerateToken(Token("", "f", "f")))
        action_sequence.eval(GenerateToken(Token("", "1", "1")))
        action_sequence.eval(ApplyRule(CloseVariadicFieldRule()))

        expected_action_sequence = ActionSequence()
        expected_action_sequence.eval(ApplyRule(funcdef))
        expected_action_sequence.eval(GenerateToken(Token("", "f", "f")))
        expected_action_sequence.eval(GenerateToken(Token(None, "1", "1")))
        expected_action_sequence.eval(ApplyRule(CloseVariadicFieldRule()))

        result = encoder.decode(encoder.encode_action(
            action_sequence, [Token(None, "1", "1")])[:-1, 1:],
            [Token(None, "1", "1")])
        self.assertEqual(expected_action_sequence.action_sequence,
                         result.action_sequence)

    def test_decode_invalid_tensor(self):
        funcdef = ExpandTreeRule(
            NodeType("def", NodeConstraint.Node, False),
            [("name",
              NodeType("value", NodeConstraint.Token, False)),
             ("body",
              NodeType("expr", NodeConstraint.Node, True))])
        expr = ExpandTreeRule(
            NodeType("expr", NodeConstraint.Node, False),
            [("op", NodeType("value", NodeConstraint.Token, False)),
             ("arg0",
              NodeType("value", NodeConstraint.Token, False)),
             ("arg1",
              NodeType("value", NodeConstraint.Token, False))])

        encoder = ActionSequenceEncoder(
            Samples([funcdef, expr],
                    [NodeType("def", NodeConstraint.Node, False),
                     NodeType("value", NodeConstraint.Token, False),
                     NodeType("expr", NodeConstraint.Node, False)],
                    [Token("", "f", "f")]),
            0)
        self.assertEqual(None,
                         encoder.decode(torch.LongTensor([[-1, -1, -1]]), []))
        self.assertEqual(None,
                         encoder.decode(torch.LongTensor([[-1, -1, 1]]), []))

    def test_encode_each_action(self):
        funcdef = ExpandTreeRule(
            NodeType("def", NodeConstraint.Node, False),
            [("name",
              NodeType("value", NodeConstraint.Token, True)),
             ("body",
              NodeType("expr", NodeConstraint.Node, True))])
        expr = ExpandTreeRule(
            NodeType("expr", NodeConstraint.Node, False),
            [("constant",
              NodeType("value", NodeConstraint.Token, True))])

        encoder = ActionSequenceEncoder(
            Samples([funcdef, expr],
                    [NodeType("def", NodeConstraint.Node, False),
                     NodeType("value", NodeConstraint.Token, True),
                     NodeType("expr", NodeConstraint.Node, False),
                     NodeType("expr", NodeConstraint.Node, True)],
                    [Token("", "f", "f"), Token("", "2", "2")]),
            0)
        action_sequence = ActionSequence()
        action_sequence.eval(ApplyRule(funcdef))
        action_sequence.eval(GenerateToken(Token("", "f", "f")))
        action_sequence.eval(GenerateToken(Token("", "1", "1")))
        action_sequence.eval(GenerateToken(Token("", "2", "2")))
        action_sequence.eval(ApplyRule(CloseVariadicFieldRule()))
        action_sequence.eval(ApplyRule(expr))
        action_sequence.eval(GenerateToken(Token("", "f", "f")))
        action_sequence.eval(ApplyRule(CloseVariadicFieldRule()))
        action_sequence.eval(ApplyRule(CloseVariadicFieldRule()))
        action = encoder.encode_each_action(
            action_sequence,
            [Token("", "1", "1"), Token("", "2", "2")],
            1)

        self.assertTrue(np.array_equal(
            np.array([
                [[1, -1, -1], [2, -1, -1]],   # funcdef
                [[-1, -1, -1], [-1, 1, -1]],  # f
                [[-1, -1, -1], [-1, -1, 0]],  # 1
                [[-1, -1, -1], [-1, 2, 1]],   # 2
                [[-1, -1, -1], [-1, -1, -1]],  # CloseVariadicField
                [[3, -1, -1], [2, -1, -1]],   # expr
                [[-1, -1, -1], [-1, 1, -1]],  # f
                [[-1, -1, -1], [-1, -1, -1]],  # CloseVariadicField
                [[-1, -1, -1], [-1, -1, -1]]  # CloseVariadicField
            ], dtype=np.long),
            action.numpy()
        ))

    def test_encode_path(self):
        funcdef = ExpandTreeRule(
            NodeType("def", NodeConstraint.Node, False),
            [("name",
              NodeType("value", NodeConstraint.Token, True)),
             ("body",
              NodeType("expr", NodeConstraint.Node, True))])
        expr = ExpandTreeRule(
            NodeType("expr", NodeConstraint.Node, False),
            [("constant",
              NodeType("value", NodeConstraint.Token, True))])

        encoder = ActionSequenceEncoder(
            Samples([funcdef, expr],
                    [NodeType("def", NodeConstraint.Node, False),
                     NodeType("value", NodeConstraint.Token, True),
                     NodeType("expr", NodeConstraint.Node, True)],
                    [Token("", "f", "f"), Token("", "2", "2")]),
            0)
        action_sequence = ActionSequence()
        action_sequence.eval(ApplyRule(funcdef))
        action_sequence.eval(GenerateToken(Token("", "f", "f")))
        action_sequence.eval(GenerateToken(Token("", "1", "1")))
        action_sequence.eval(GenerateToken(Token("", "2", "2")))
        action_sequence.eval(ApplyRule(CloseVariadicFieldRule()))
        action_sequence.eval(ApplyRule(expr))
        action_sequence.eval(GenerateToken(Token("", "f", "f")))
        action_sequence.eval(ApplyRule(CloseVariadicFieldRule()))
        action_sequence.eval(ApplyRule(CloseVariadicFieldRule()))
        path = encoder.encode_path(action_sequence, 2)

        self.assertTrue(np.array_equal(
            np.array([
                [-1, -1],  # funcdef
                [2, -1],  # f
                [2, -1],  # 1
                [2, -1],  # 2
                [2, -1],  # CloseVariadicField
                [2, -1],  # expr
                [3, 2],  # f
                [3, 2],  # CloseVariadicField
                [2, -1],  # CloseVariadicField
            ], dtype=np.long),
            path.numpy()
        ))
        path = encoder.encode_path(action_sequence, 1)
        self.assertTrue(np.array_equal(
            np.array([
                [-1],  # funcdef
                [2],  # f
                [2],  # 1
                [2],  # 2
                [2],  # CloseVariadicField
                [2],  # expr
                [3],  # f
                [3],  # CloseVariadicField
                [2],  # CloseVariadicField
            ], dtype=np.long),
            path.numpy()
        ))


if __name__ == "__main__":
    unittest.main()
