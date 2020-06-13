import torch
import unittest
import numpy as np

from mlprogram.actions \
    import ExpandTreeRule, NodeType, NodeConstraint, CloseNode, \
    ApplyRule, GenerateToken, ActionOptions, CloseVariadicFieldRule
from mlprogram.actions import ActionSequence
from mlprogram.encoders import Samples, ActionSequenceEncoder


class TestEncoder(unittest.TestCase):
    def test_reserved_labels(self):
        encoder = ActionSequenceEncoder(
            Samples([], [], [], ActionOptions(True, True)), 0)
        self.assertEqual(2, len(encoder._rule_encoder.vocab))
        self.assertEqual(2, len(encoder._token_encoder.vocab))

        encoder = ActionSequenceEncoder(
            Samples([], [], [], ActionOptions(False, True)), 0)
        self.assertEqual(1, len(encoder._rule_encoder.vocab))
        self.assertEqual(2, len(encoder._token_encoder.vocab))

        encoder = ActionSequenceEncoder(
            Samples([], [], [], ActionOptions(True, False)), 0)
        self.assertEqual(2, len(encoder._rule_encoder.vocab))
        self.assertEqual(1, len(encoder._token_encoder.vocab))

    def test_encode_action(self):
        funcdef = ExpandTreeRule(NodeType("def", NodeConstraint.Node),
                                 [("name",
                                   NodeType("value", NodeConstraint.Token)),
                                  ("body",
                                   NodeType("expr", NodeConstraint.Variadic))])
        expr = ExpandTreeRule(NodeType("expr", NodeConstraint.Node),
                              [("op", NodeType("value", NodeConstraint.Token)),
                               ("arg0",
                                NodeType("value", NodeConstraint.Token)),
                               ("arg1",
                                NodeType("value", NodeConstraint.Token))])

        encoder = ActionSequenceEncoder(
            Samples([funcdef, expr],
                    [NodeType("def", NodeConstraint.Node),
                     NodeType("value", NodeConstraint.Token),
                     NodeType("expr", NodeConstraint.Node)],
                    ["f", "2"],
                    ActionOptions(True, True)),
            0)
        evaluator = ActionSequence()
        evaluator.eval(ApplyRule(funcdef))
        evaluator.eval(GenerateToken("f"))
        evaluator.eval(GenerateToken("1"))
        evaluator.eval(GenerateToken("2"))
        evaluator.eval(GenerateToken(CloseNode()))
        action = encoder.encode_action(evaluator, ["1", "2"])

        self.assertTrue(np.array_equal(
            [
                [-1, 2, -1, -1],
                [2, -1, 2, -1],
                [2, -1, -1, 0],
                [2, -1, 3, 1],
                [2, -1, 1, -1],
                [3, -1, -1, -1]
            ],
            action.numpy()
        ))

    def test_encode_parent(self):
        funcdef = ExpandTreeRule(NodeType("def", NodeConstraint.Node),
                                 [("name",
                                   NodeType("value", NodeConstraint.Token)),
                                  ("body",
                                   NodeType("expr", NodeConstraint.Variadic))])
        expr = ExpandTreeRule(NodeType("expr", NodeConstraint.Node),
                              [("op", NodeType("value", NodeConstraint.Token)),
                               ("arg0",
                                NodeType("value", NodeConstraint.Token)),
                               ("arg1",
                                NodeType("value", NodeConstraint.Token))])

        encoder = ActionSequenceEncoder(
            Samples([funcdef, expr],
                    [NodeType("def", NodeConstraint.Node),
                     NodeType("value", NodeConstraint.Token),
                     NodeType("expr", NodeConstraint.Node)],
                    ["f", "2"],
                    ActionOptions(True, True)),
            0)
        evaluator = ActionSequence()
        evaluator.eval(ApplyRule(funcdef))
        evaluator.eval(GenerateToken("f"))
        evaluator.eval(GenerateToken("1"))
        evaluator.eval(GenerateToken("2"))
        evaluator.eval(GenerateToken(CloseNode()))
        parent = encoder.encode_parent(evaluator)

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
        funcdef = ExpandTreeRule(NodeType("def", NodeConstraint.Node),
                                 [("name",
                                   NodeType("value", NodeConstraint.Token)),
                                  ("body",
                                   NodeType("expr", NodeConstraint.Variadic))])
        expr = ExpandTreeRule(NodeType("expr", NodeConstraint.Node),
                              [("op", NodeType("value", NodeConstraint.Token)),
                               ("arg0",
                                NodeType("value", NodeConstraint.Token)),
                               ("arg1",
                                NodeType("value", NodeConstraint.Token))])

        encoder = ActionSequenceEncoder(
            Samples([funcdef, expr],
                    [NodeType("def", NodeConstraint.Node),
                     NodeType("value", NodeConstraint.Token),
                     NodeType("expr", NodeConstraint.Node)],
                    ["f", "2"],
                    ActionOptions(True, True)),
            0)
        evaluator = ActionSequence()
        evaluator.eval(ApplyRule(funcdef))
        evaluator.eval(GenerateToken("f"))
        evaluator.eval(GenerateToken("1"))
        d, m = encoder.encode_tree(evaluator)

        self.assertTrue(np.array_equal(
            [0, 1, 1], d.numpy()
        ))
        self.assertTrue(np.array_equal(
            [[0, 1, 1], [0, 0, 0], [0, 0, 0]],
            m.numpy()
        ))

    def test_encode_empty_sequence(self):
        funcdef = ExpandTreeRule(NodeType("def", NodeConstraint.Node),
                                 [("name",
                                   NodeType("value", NodeConstraint.Token)),
                                  ("body",
                                   NodeType("expr", NodeConstraint.Variadic))])
        expr = ExpandTreeRule(NodeType("expr", NodeConstraint.Node),
                              [("op", NodeType("value", NodeConstraint.Token)),
                               ("arg0",
                                NodeType("value", NodeConstraint.Token)),
                               ("arg1",
                                NodeType("value", NodeConstraint.Token))])

        encoder = ActionSequenceEncoder(
            Samples([funcdef, expr],
                    [NodeType("def", NodeConstraint.Node),
                     NodeType("value", NodeConstraint.Token),
                     NodeType("expr", NodeConstraint.Node)],
                    ["f"],
                    ActionOptions(True, True)),
            0)
        evaluator = ActionSequence()
        action = encoder.encode_action(evaluator, ["1"])
        parent = encoder.encode_parent(evaluator)
        d, m = encoder.encode_tree(evaluator)

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
        funcdef = ExpandTreeRule(NodeType("def", NodeConstraint.Node),
                                 [("name",
                                   NodeType("value", NodeConstraint.Token)),
                                  ("body",
                                   NodeType("expr", NodeConstraint.Variadic))])
        expr = ExpandTreeRule(NodeType("expr", NodeConstraint.Node),
                              [("op", NodeType("value", NodeConstraint.Token)),
                               ("arg0",
                                NodeType("value", NodeConstraint.Token)),
                               ("arg1",
                                NodeType("value", NodeConstraint.Token))])

        encoder = ActionSequenceEncoder(
            Samples([funcdef, expr],
                    [NodeType("def", NodeConstraint.Node),
                     NodeType("value", NodeConstraint.Token),
                     NodeType("expr", NodeConstraint.Node)],
                    ["f"],
                    ActionOptions(True, True)),
            0)
        evaluator = ActionSequence()
        evaluator.eval(ApplyRule(funcdef))
        evaluator.eval(GenerateToken("f"))
        evaluator.eval(GenerateToken("1"))
        evaluator.eval(GenerateToken(CloseNode()))

        self.assertEqual(None, encoder.encode_action(evaluator, ["2"]))

    def test_encode_completed_sequence(self):
        none = ExpandTreeRule(NodeType("value", NodeConstraint.Node),
                              [])
        encoder = ActionSequenceEncoder(
            Samples([none],
                    [NodeType("value", NodeConstraint.Node)],
                    ["f"],
                    ActionOptions(True, True)),
            0)
        evaluator = ActionSequence()
        evaluator.eval(ApplyRule(none))
        action = encoder.encode_action(evaluator, ["1"])
        parent = encoder.encode_parent(evaluator)

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
        funcdef = ExpandTreeRule(NodeType("def", NodeConstraint.Node),
                                 [("name",
                                   NodeType("value", NodeConstraint.Token)),
                                  ("body",
                                   NodeType("expr", NodeConstraint.Variadic))])
        expr = ExpandTreeRule(NodeType("expr", NodeConstraint.Node),
                              [("op", NodeType("value", NodeConstraint.Token)),
                               ("arg0",
                                NodeType("value", NodeConstraint.Token)),
                               ("arg1",
                                NodeType("value", NodeConstraint.Token))])

        encoder = ActionSequenceEncoder(
            Samples([funcdef, expr],
                    [NodeType("def", NodeConstraint.Node),
                     NodeType("value", NodeConstraint.Token),
                     NodeType("expr", NodeConstraint.Node)],
                    ["f"],
                    ActionOptions(True, True)),
            0)
        evaluator = ActionSequence()
        evaluator.eval(ApplyRule(funcdef))
        evaluator.eval(GenerateToken("f"))
        evaluator.eval(GenerateToken("1"))
        evaluator.eval(GenerateToken(CloseNode()))

        result = encoder.decode(encoder.encode_action(
            evaluator, ["1"])[:-1, 1:], ["1"])
        self.assertEqual(evaluator.action_sequence, result.action_sequence)

    def test_decode_invalid_tensor(self):
        funcdef = ExpandTreeRule(NodeType("def", NodeConstraint.Node),
                                 [("name",
                                   NodeType("value", NodeConstraint.Token)),
                                  ("body",
                                   NodeType("expr", NodeConstraint.Variadic))])
        expr = ExpandTreeRule(NodeType("expr", NodeConstraint.Node),
                              [("op", NodeType("value", NodeConstraint.Token)),
                               ("arg0",
                                NodeType("value", NodeConstraint.Token)),
                               ("arg1",
                                NodeType("value", NodeConstraint.Token))])

        encoder = ActionSequenceEncoder(
            Samples([funcdef, expr],
                    [NodeType("def", NodeConstraint.Node),
                     NodeType("value", NodeConstraint.Token),
                     NodeType("expr", NodeConstraint.Node)],
                    ["f"],
                    ActionOptions(True, True)),
            0)
        self.assertEqual(None,
                         encoder.decode(torch.LongTensor([[-1, -1, -1]]), []))
        self.assertEqual(None,
                         encoder.decode(torch.LongTensor([[-1, -1, 1]]), []))

    def test_encode_each_action(self):
        funcdef = ExpandTreeRule(NodeType("def", NodeConstraint.Node),
                                 [("name",
                                   NodeType("value", NodeConstraint.Token)),
                                  ("body",
                                   NodeType("expr", NodeConstraint.Variadic))])
        expr = ExpandTreeRule(NodeType("expr", NodeConstraint.Node),
                              [("constant",
                                NodeType("value", NodeConstraint.Token))])

        encoder = ActionSequenceEncoder(
            Samples([funcdef, expr],
                    [NodeType("def", NodeConstraint.Node),
                     NodeType("value", NodeConstraint.Token),
                     NodeType("expr", NodeConstraint.Node)],
                    ["f", "2"],
                    ActionOptions(True, True)),
            0)
        evaluator = ActionSequence()
        evaluator.eval(ApplyRule(funcdef))
        evaluator.eval(GenerateToken("f"))
        evaluator.eval(GenerateToken("1"))
        evaluator.eval(GenerateToken("2"))
        evaluator.eval(GenerateToken(CloseNode()))
        evaluator.eval(ApplyRule(expr))
        evaluator.eval(GenerateToken("f"))
        evaluator.eval(GenerateToken(CloseNode()))
        evaluator.eval(ApplyRule(CloseVariadicFieldRule()))
        action = encoder.encode_each_action(evaluator, ["1", "2"], 1)

        self.assertTrue(np.array_equal(
            np.array([
                [[1, -1, -1], [2, -1, -1]],   # funcdef
                [[-1, -1, -1], [-1, 2, -1]],  # f
                [[-1, -1, -1], [-1, -1, 0]],  # 1
                [[-1, -1, -1], [-1, 3, 1]],   # 2
                [[-1, -1, -1], [-1, 1, -1]],  # CloseNode
                [[3, -1, -1], [2, -1, -1]],   # expr
                [[-1, -1, -1], [-1, 2, -1]],  # f
                [[-1, -1, -1], [-1, 1, -1]],  # CloseNode
                [[-1, -1, -1], [-1, -1, -1]]  # CloseVariadicField
            ], dtype=np.long),
            action.numpy()
        ))

    def test_encode_path(self):
        funcdef = ExpandTreeRule(NodeType("def", NodeConstraint.Node),
                                 [("name",
                                   NodeType("value", NodeConstraint.Token)),
                                  ("body",
                                   NodeType("expr", NodeConstraint.Variadic))])
        expr = ExpandTreeRule(NodeType("expr", NodeConstraint.Node),
                              [("constant",
                                NodeType("value", NodeConstraint.Token))])

        encoder = ActionSequenceEncoder(
            Samples([funcdef, expr],
                    [NodeType("def", NodeConstraint.Node),
                     NodeType("value", NodeConstraint.Token),
                     NodeType("expr", NodeConstraint.Node)],
                    ["f", "2"],
                    ActionOptions(True, True)),
            0)
        evaluator = ActionSequence()
        evaluator.eval(ApplyRule(funcdef))
        evaluator.eval(GenerateToken("f"))
        evaluator.eval(GenerateToken("1"))
        evaluator.eval(GenerateToken("2"))
        evaluator.eval(GenerateToken(CloseNode()))
        evaluator.eval(ApplyRule(expr))
        evaluator.eval(GenerateToken("f"))
        evaluator.eval(GenerateToken(CloseNode()))
        evaluator.eval(ApplyRule(CloseVariadicFieldRule()))
        path = encoder.encode_path(evaluator, 2)

        self.assertTrue(np.array_equal(
            np.array([
                [-1, -1],  # funcdef
                [2, -1],  # f
                [2, -1],  # 1
                [2, -1],  # 2
                [2, -1],  # CloseNode
                [2, -1],  # expr
                [3, 2],  # f
                [3, 2],  # CloseNode
                [2, -1],  # CloseVariadicField
            ], dtype=np.long),
            path.numpy()
        ))
        path = encoder.encode_path(evaluator, 1)
        self.assertTrue(np.array_equal(
            np.array([
                [-1],  # funcdef
                [2],  # f
                [2],  # 1
                [2],  # 2
                [2],  # CloseNode
                [2],  # expr
                [3],  # f
                [3],  # CloseNode
                [2],  # CloseVariadicField
            ], dtype=np.long),
            path.numpy()
        ))

    def test_remove_variadic_node_types(self):
        self.assertEqual(
            [NodeType("t1", NodeConstraint.Node),
             NodeType("t2", NodeConstraint.Token)],
            ActionSequenceEncoder.remove_variadic_node_types(
                [NodeType("t1", NodeConstraint.Node),
                 NodeType("t2", NodeConstraint.Token)]))
        self.assertEqual(
            [NodeType("t1", NodeConstraint.Node),
             NodeType("t2", NodeConstraint.Token)],
            ActionSequenceEncoder.remove_variadic_node_types(
                [NodeType("t1", NodeConstraint.Variadic),
                 NodeType("t2", NodeConstraint.Token)]))
        self.assertEqual(
            [NodeType("t1", NodeConstraint.Node),
             NodeType("t2", NodeConstraint.Token)],
            ActionSequenceEncoder.remove_variadic_node_types(
                [NodeType("t1", NodeConstraint.Variadic),
                 NodeType("t2", NodeConstraint.Token),
                 NodeType("t1", NodeConstraint.Node)]))


if __name__ == "__main__":
    unittest.main()
