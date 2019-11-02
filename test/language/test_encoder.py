import torch
import unittest
import numpy as np

from nl2code.language.action import ApplyRule, GenerateToken
from nl2code.language.action import ExpandTreeRule, NodeType, NodeConstraint
from nl2code.language.action import CloseNode

from nl2code.language.evaluator import Evaluator
from nl2code.language.encoder import Encoder


class TestEncoder(unittest.TestCase):
    def test_encode(self):
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

        encoder = Encoder([funcdef, expr],
                          [NodeType("def", NodeConstraint.Node),
                           NodeType("value", NodeConstraint.Token),
                           NodeType("expr", NodeConstraint.Node)],
                          ["f", "2"],
                          0)
        evaluator = Evaluator()
        evaluator.eval(ApplyRule(funcdef))
        evaluator.eval(GenerateToken("f"))
        evaluator.eval(GenerateToken("1"))
        evaluator.eval(GenerateToken("2"))
        evaluator.eval(GenerateToken(CloseNode()))
        encoded_tensor = encoder.encode(evaluator, ["1", "2"])

        self.assertTrue(np.array_equal(
            [
                [-1, -1, -1],
                [2, 2, 0],
                [2, 2, 0],
                [2, 2, 0],
                [2, 2, 0],
                [3, 2, 0]
            ],
            encoded_tensor.action.numpy()
        ))
        self.assertTrue(np.array_equal(
            [
                [-1, -1, -1],
                [2, -1, -1],
                [-1, 2, -1],
                [-1, -1, 0],
                [-1, 3, 1],
                [-1, 1, -1]
            ],
            encoded_tensor.previous_action.numpy()
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

        encoder = Encoder([funcdef, expr],
                          [NodeType("def", NodeConstraint.Node),
                           NodeType("value", NodeConstraint.Token),
                           NodeType("expr", NodeConstraint.Node)],
                          ["f"],
                          0)
        evaluator = Evaluator()
        encoded_tensor = encoder.encode(evaluator, ["1"])

        self.assertTrue(np.array_equal(
            [
                [-1, -1, -1]
            ],
            encoded_tensor.action.numpy()
        ))
        self.assertTrue(np.array_equal(
            [
                [-1, -1, -1]
            ],
            encoded_tensor.previous_action.numpy()
        ))

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

        encoder = Encoder([funcdef, expr],
                          [NodeType("def", NodeConstraint.Node),
                           NodeType("value", NodeConstraint.Token),
                           NodeType("expr", NodeConstraint.Node)],
                          ["f"],
                          0)
        evaluator = Evaluator()
        evaluator.eval(ApplyRule(funcdef))
        evaluator.eval(GenerateToken("f"))
        evaluator.eval(GenerateToken("1"))
        evaluator.eval(GenerateToken(CloseNode()))

        self.assertEqual(None, encoder.encode(evaluator, ["2"]))

    def test_encode_completed_sequence(self):
        none = ExpandTreeRule(NodeType("value", NodeConstraint.Node),
                              [])
        encoder = Encoder([none],
                          [NodeType("value", NodeConstraint.Node)],
                          ["f"],
                          0)
        evaluator = Evaluator()
        evaluator.eval(ApplyRule(none))
        encoded_tensor = encoder.encode(evaluator, ["1"])

        self.assertTrue(np.array_equal(
            [
                [-1, -1, -1],
                [-1, -1, -1]
            ],
            encoded_tensor.action.numpy()
        ))
        self.assertTrue(np.array_equal(
            [
                [-1, -1, -1],
                [2, -1, -1],
            ],
            encoded_tensor.previous_action.numpy()
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

        encoder = Encoder([funcdef, expr],
                          [NodeType("def", NodeConstraint.Node),
                           NodeType("value", NodeConstraint.Token),
                           NodeType("expr", NodeConstraint.Node)],
                          ["f"],
                          0)
        evaluator = Evaluator()
        evaluator.eval(ApplyRule(funcdef))
        evaluator.eval(GenerateToken("f"))
        evaluator.eval(GenerateToken("1"))
        evaluator.eval(GenerateToken(CloseNode()))

        result = encoder.decode(encoder.encode(
            evaluator, ["1"]).previous_action[1:, :], ["1"])
        self.assertEqual(evaluator.action_sequence, result)

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

        encoder = Encoder([funcdef, expr],
                          [NodeType("def", NodeConstraint.Node),
                           NodeType("value", NodeConstraint.Token),
                           NodeType("expr", NodeConstraint.Node)],
                          ["f"],
                          0)
        self.assertEqual(None,
                         encoder.decode(torch.LongTensor([[-1, -1, -1]]), []))
        self.assertEqual(None,
                         encoder.decode(torch.LongTensor([[-1, -1, 1]]), []))

    def test_remove_variadic_node_types(self):
        self.assertEqual(
            [NodeType("t1", NodeConstraint.Node),
             NodeType("t2", NodeConstraint.Token)],
            Encoder.remove_variadic_node_types(
                [NodeType("t1", NodeConstraint.Node),
                 NodeType("t2", NodeConstraint.Token)]))
        self.assertEqual(
            [NodeType("t1", NodeConstraint.Node),
             NodeType("t2", NodeConstraint.Token)],
            Encoder.remove_variadic_node_types(
                [NodeType("t1", NodeConstraint.Variadic),
                 NodeType("t2", NodeConstraint.Token)]))
        self.assertEqual(
            [NodeType("t1", NodeConstraint.Node),
             NodeType("t2", NodeConstraint.Token)],
            Encoder.remove_variadic_node_types(
                [NodeType("t1", NodeConstraint.Variadic),
                 NodeType("t2", NodeConstraint.Token),
                 NodeType("t1", NodeConstraint.Node)]))


if __name__ == "__main__":
    unittest.main()
