import torch
import unittest
import numpy as np

from nl2code import BeamSearchSynthesizer, Progress, Candidate
from nl2code.language.ast import Node, Field, Leaf
from nl2code.language.action import NodeConstraint, NodeType
from nl2code.language.action import ExpandTreeRule, CloseVariadicFieldRule
from nl2code.language.action import ApplyRule, GenerateToken, CloseNode
from nl2code.language.encoder import Encoder


class MockPredictor:
    def __init__(self, rule_preds, token_preds, copy_preds, histories, hs, cs):
        self.rule_preds = rule_preds
        self.token_preds = token_preds
        self.copy_preds = copy_preds
        self.histories = histories
        self.hs = hs
        self.cs = cs
        self.arguments = []
        self.hidden_size = 1

    def parameters(self):
        return [torch.zeros(1, 1)]

    def __call__(self, query, action, prev_action,
                 hist, states):
        i = len(self.arguments)
        self.arguments.append((query, action, prev_action, hist, states))
        return (self.rule_preds[i], self.token_preds[i], self.copy_preds[i],
                self.histories[i], (self.hs[i], self.cs[i]))


X = NodeType("X", NodeConstraint.Node)
Y = NodeType("Y", NodeConstraint.Node)
Y_list = NodeType("Y", NodeConstraint.Variadic)
Ysub = NodeType("Ysub", NodeConstraint.Node)
Str = NodeType("Str", NodeConstraint.Token)


class TestBeamSearchSynthesizer(unittest.TestCase):
    def assertSameProgress(self, expected: Progress, actual: Progress):
        self.assertEqual(expected.id, actual.id)
        self.assertEqual(expected.parent, actual.parent)
        self.assertAlmostEqual(expected.score, actual.score, delta=1e-5)
        self.assertEqual(expected.action, actual.action)
        self.assertEqual(expected.is_complete, actual.is_complete)

    def assertSameCandidate(self, expected: Candidate, actual: Candidate):
        self.assertAlmostEqual(expected.score, actual.score, delta=1e-5)
        self.assertEqual(expected.ast, actual.ast)

    def test_apply_rule_generation(self):
        XtoY = ExpandTreeRule(X, [("value", Y)])
        YsubtoNone = ExpandTreeRule(Ysub, [])
        encoder = Encoder([XtoY, YsubtoNone], [X, Y, Ysub], ["foo"], 0)

        def is_subtype(arg0, arg1):
            if arg0 == arg1:
                return True
            if arg0 == Ysub and arg1 == Y:
                return True
            return False

        # Prepare mock probabilities
        rule0 = torch.FloatTensor([[[0.0, 0.0, 0.9, 0.1]]])
        token0 = torch.FloatTensor([[[0.0, 0.0, 0.0]]])
        copy0 = torch.FloatTensor([[[0.0]]])
        history0 = torch.FloatTensor(1, 1, 1)
        h0 = torch.FloatTensor(1, 1)
        c0 = torch.FloatTensor(1, 1)
        rule1 = torch.FloatTensor([[[0.0, 0.0, 0.0, 1.0]]])
        token1 = torch.FloatTensor([[[0.0, 0.0, 0.0]]])
        copy1 = torch.FloatTensor([[[0.0]]])
        history1 = torch.FloatTensor(2, 1, 1)
        h1 = torch.FloatTensor(1, 1)
        c1 = torch.FloatTensor(1, 1)
        predictor = MockPredictor(
            [rule0, rule1], [token0, token1], [copy0, copy1],
            [history0, history1], [h0, h1], [c0, c1])

        synthesizer = BeamSearchSynthesizer(2,
                                            predictor, encoder, is_subtype)
        results = synthesizer.synthesize(["test"], torch.FloatTensor(1, 1))
        """
        [] -> [XtoY] -> [XtoY, YsubtoNone] (Complete)
           -> [YsubtoNone] (Complete)
        """
        progress = results[0]
        candidates = results[1]
        self.assertEqual(2, len(progress))
        self.assertSameProgress(
            Progress(1, 0, np.log(0.9), ApplyRule(XtoY), False),
            progress[0][0]
        )
        self.assertSameProgress(
            Progress(2, 0, np.log(0.1), ApplyRule(YsubtoNone), True),
            progress[0][1]
        )
        self.assertSameProgress(
            Progress(3, 1, np.log(0.9), ApplyRule(YsubtoNone), True),
            progress[1][0]
        )

        self.assertEqual(2, len(candidates))
        self.assertSameCandidate(
            Candidate(np.log(0.1), Node("Ysub", [])), candidates[0]
        )
        self.assertSameCandidate(
            Candidate(np.log(0.9),
                      Node("X", [Field("value", "Y", Node("Ysub", []))])),
            candidates[1]
        )

    def test_variadic_fields_generation(self):
        XtoY = ExpandTreeRule(X, [("value", Y_list)])
        YsubtoNone = ExpandTreeRule(Ysub, [])
        encoder = Encoder([XtoY, YsubtoNone], [X, Y, Ysub], ["foo"], 0)

        def is_subtype(arg0, arg1):
            if arg0 == arg1:
                return True
            if arg0 == Ysub and arg1 == Y_list:
                return True
            return False

        # Prepare mock probabilities
        rule0 = torch.FloatTensor([[[0.0, 0.0, 0.9, 0.1]]])
        token0 = torch.FloatTensor([[[0.0, 0.0, 0.0]]])
        copy0 = torch.FloatTensor([[[0.0]]])
        history0 = torch.FloatTensor(1, 1, 1)
        h0 = torch.FloatTensor(1, 1)
        c0 = torch.FloatTensor(1, 1)
        rule1 = torch.FloatTensor([[[0.0, 0.1, 0.0, 0.9]]])
        token1 = torch.FloatTensor([[[0.0, 0.0, 0.0]]])
        copy1 = torch.FloatTensor([[[0.0]]])
        history1 = torch.FloatTensor(2, 1, 1)
        h1 = torch.FloatTensor(1, 1)
        c1 = torch.FloatTensor(1, 1)
        rule2 = torch.FloatTensor([[[0.0, 0.9, 0.0, 0.1]]])
        token2 = torch.FloatTensor([[[0.0, 0.0, 0.0]]])
        copy2 = torch.FloatTensor([[[0.0]]])
        history2 = torch.FloatTensor(3, 1, 1)
        h2 = torch.FloatTensor(1, 1)
        c2 = torch.FloatTensor(1, 1)
        predictor = MockPredictor(
            [rule0, rule1, rule2], [token0, token1, token2],
            [copy0, copy1, copy2],
            [history0, history1, history2], [h0, h1, h2], [c0, c1, c2])

        synthesizer = BeamSearchSynthesizer(3,
                                            predictor, encoder, is_subtype)
        results = synthesizer.synthesize(["test"], torch.FloatTensor(1, 1))
        """
        [] -> [XtoY] -> [XtoY, YsubtoNone] -> [XtoY, YsubtoNone, Close]
           -> [YsubtoNone] (Complete)
        """
        progress = results[0]
        candidates = results[1]
        self.assertEqual(3, len(progress))
        self.assertSameProgress(
            Progress(1, 0, np.log(0.9), ApplyRule(XtoY), False),
            progress[0][0]
        )
        self.assertSameProgress(
            Progress(2, 0, np.log(0.1), ApplyRule(YsubtoNone), True),
            progress[0][1]
        )
        self.assertSameProgress(
            Progress(3, 1, np.log(0.9) + np.log(0.9),
                     ApplyRule(YsubtoNone), False),
            progress[1][0]
        )
        self.assertSameProgress(
            Progress(4, 1, np.log(0.9) + np.log(0.1),
                     ApplyRule(CloseVariadicFieldRule()), True),
            progress[1][1]
        )
        self.assertSameProgress(
            Progress(5, 3, np.log(0.9) + np.log(0.9) + np.log(0.9),
                     ApplyRule(CloseVariadicFieldRule()), True),
            progress[2][0]
        )

        self.assertEqual(3, len(candidates))
        self.assertSameCandidate(
            Candidate(np.log(0.1), Node("Ysub", [])), candidates[0]
        )
        self.assertSameCandidate(
            Candidate(np.log(0.9) + np.log(0.1),
                      Node("X", [Field("value", "Y", [])])),
            candidates[1]
        )
        self.assertSameCandidate(
            Candidate(np.log(0.9) + np.log(0.9) + np.log(0.9),
                      Node("X", [Field("value", "Y", [Node("Ysub", [])])])),
            candidates[2]
        )

    def test_token_generation(self):
        XtoStr = ExpandTreeRule(X, [("value", Str)])
        encoder = Encoder([XtoStr], [X, Str], ["foo"], 0)

        def is_subtype(arg0, arg1):
            if arg0 == arg1:
                return True
            if arg0 == Ysub and arg1 == Y_list:
                return True
            return False

        # Prepare mock probabilities
        rule0 = torch.FloatTensor([[[0.0, 0.0, 1.0]]])
        token0 = torch.FloatTensor([[[0.0, 0.0, 0.0]]])
        copy0 = torch.FloatTensor([[[0.0]]])
        history0 = torch.FloatTensor(1, 1, 1)
        h0 = torch.FloatTensor(1, 1)
        c0 = torch.FloatTensor(1, 1)
        rule1 = torch.FloatTensor([[[0.0, 0.0, 0.0]]])
        token1 = torch.FloatTensor([[[0.0, 0.1, 0.9]]])
        copy1 = torch.FloatTensor([[[0.0]]])
        history1 = torch.FloatTensor(2, 1, 1)
        h1 = torch.FloatTensor(1, 1)
        c1 = torch.FloatTensor(1, 1)
        rule2 = torch.FloatTensor([[[0.0, 0.0, 0.0]]])
        token2 = torch.FloatTensor([[[0.0, 1.0, 0.0]]])
        copy2 = torch.FloatTensor([[[0.0]]])
        history2 = torch.FloatTensor(3, 1, 1)
        h2 = torch.FloatTensor(1, 1)
        c2 = torch.FloatTensor(1, 1)
        predictor = MockPredictor(
            [rule0, rule1, rule2], [token0, token1, token2],
            [copy0, copy1, copy2],
            [history0, history1, history2], [h0, h1, h2], [c0, c1, c2])

        synthesizer = BeamSearchSynthesizer(2,
                                            predictor, encoder, is_subtype)
        results = synthesizer.synthesize(["test"], torch.FloatTensor(1, 1))
        """
        [] -> [XtoStr] -> "foo" -> CloseNode (Complete)
                       -> CloseNode (Complete)
        """
        progress = results[0]
        candidates = results[1]
        self.assertEqual(3, len(progress))
        self.assertSameProgress(
            Progress(1, 0, np.log(1.0), ApplyRule(XtoStr), False),
            progress[0][0]
        )
        self.assertSameProgress(
            Progress(2, 1, np.log(0.9), GenerateToken("foo"), False),
            progress[1][0]
        )
        self.assertSameProgress(
            Progress(3, 1, np.log(0.1),
                     GenerateToken(CloseNode()), True),
            progress[1][1]
        )
        self.assertSameProgress(
            Progress(4, 2, np.log(0.9) + np.log(1.0),
                     GenerateToken(CloseNode()), True),
            progress[2][0]
        )

        self.assertEqual(2, len(candidates))
        self.assertSameCandidate(
            Candidate(np.log(0.1),
                      Node("X", [Field("value", "Str", Leaf("Str", ""))])),
            candidates[0]
        )
        self.assertSameCandidate(
            Candidate(np.log(0.9),
                      Node("X", [Field("value", "Str", Leaf("Str", "foo"))])),
            candidates[1]
        )

    def test_copy_action_generation(self):
        XtoStr = ExpandTreeRule(X, [("value", Str)])
        encoder = Encoder([XtoStr], [X, Str], ["xxx"], 0)

        def is_subtype(arg0, arg1):
            if arg0 == arg1:
                return True
            if arg0 == Ysub and arg1 == Y_list:
                return True
            return False

        # Prepare mock probabilities
        rule0 = torch.FloatTensor([[[0.0, 0.0, 1.0]]])
        token0 = torch.FloatTensor([[[0.0, 0.0, 0.0]]])
        copy0 = torch.FloatTensor([[[0.0]]])
        history0 = torch.FloatTensor(1, 1, 1)
        h0 = torch.FloatTensor(1, 1)
        c0 = torch.FloatTensor(1, 1)
        rule1 = torch.FloatTensor([[[0.0, 0.0, 0.0]]])
        token1 = torch.FloatTensor([[[0.0, 0.1, 0.0]]])
        copy1 = torch.FloatTensor([[[0.9]]])
        history1 = torch.FloatTensor(2, 1, 1)
        h1 = torch.FloatTensor(1, 1)
        c1 = torch.FloatTensor(1, 1)
        rule2 = torch.FloatTensor([[[0.0, 0.0, 0.0]]])
        token2 = torch.FloatTensor([[[0.0, 1.0, 0.0]]])
        copy2 = torch.FloatTensor([[[0.0]]])
        history2 = torch.FloatTensor(3, 1, 1)
        h2 = torch.FloatTensor(1, 1)
        c2 = torch.FloatTensor(1, 1)
        predictor = MockPredictor(
            [rule0, rule1, rule2], [token0, token1, token2],
            [copy0, copy1, copy2],
            [history0, history1, history2], [h0, h1, h2], [c0, c1, c2])

        synthesizer = BeamSearchSynthesizer(2,
                                            predictor, encoder, is_subtype)
        results = synthesizer.synthesize(["foo"], torch.FloatTensor(1, 1))
        """
        [] -> [XtoStr] -> "foo" -> CloseNode (Complete)
                       -> CloseNode (Complete)
        """
        progress = results[0]
        candidates = results[1]
        self.assertEqual(3, len(progress))
        self.assertSameProgress(
            Progress(1, 0, np.log(1.0), ApplyRule(XtoStr), False),
            progress[0][0]
        )
        self.assertSameProgress(
            Progress(2, 1, np.log(0.9), GenerateToken("foo"), False),
            progress[1][0]
        )
        self.assertSameProgress(
            Progress(3, 1, np.log(0.1),
                     GenerateToken(CloseNode()), True),
            progress[1][1]
        )
        self.assertSameProgress(
            Progress(4, 2, np.log(0.9) + np.log(1.0),
                     GenerateToken(CloseNode()), True),
            progress[2][0]
        )

        self.assertEqual(2, len(candidates))
        self.assertSameCandidate(
            Candidate(np.log(0.1),
                      Node("X", [Field("value", "Str", Leaf("Str", ""))])),
            candidates[0]
        )
        self.assertSameCandidate(
            Candidate(np.log(0.9),
                      Node("X", [Field("value", "Str", Leaf("Str", "foo"))])),
            candidates[1]
        )


if __name__ == "__main__":
    unittest.main()
