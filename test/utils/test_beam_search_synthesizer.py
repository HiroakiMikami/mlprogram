import unittest
import numpy as np
from math import log

from mlprogram.synthesizer \
    import BeamSearchSynthesizer, Progress, Candidate, \
    LazyLogProbability
from mlprogram.ast.ast import Node, Field, Leaf, Root
from mlprogram.ast.action \
    import NodeConstraint, NodeType, ExpandTreeRule, CloseVariadicFieldRule, \
    ApplyRule, GenerateToken, CloseNode, ActionOptions


class MockBeamSearchSynthesizer(BeamSearchSynthesizer):
    def __init__(self, beam_size, is_subtype, rule_preds, token_preds,
                 options=None):
        if options is None:
            super(MockBeamSearchSynthesizer, self).__init__(
                beam_size, is_subtype)
        else:
            super(MockBeamSearchSynthesizer, self).__init__(
                beam_size, is_subtype, options=options)
        self.rule_preds = rule_preds
        self.token_preds = token_preds
        self.arguments = []

    def initialize(self, query: str):
        return 0

    def batch_update(self, hs):
        i = len(self.arguments)
        self.arguments.append(([h.state for h in hs]))
        rule_preds = self.rule_preds[i]
        token_preds = self.token_preds[i]
        return [(h.state + 1, LazyLogProbability(
            lambda: rule_preds[i], lambda: token_preds[i]
        )) for i, h in enumerate(hs)]


X = NodeType("X", NodeConstraint.Node)
Y = NodeType("Y", NodeConstraint.Node)
Y_list = NodeType("Y", NodeConstraint.Variadic)
Ysub = NodeType("Ysub", NodeConstraint.Node)
Str = NodeType("Str", NodeConstraint.Token)


def is_subtype(arg0, arg1):
    if arg0 == arg1:
        return True
    if arg0 == "Ysub" and arg1 == "Y":
        return True
    return False


Zero = log(1e-5)


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

        # Prepare mock probabilities
        rule0 = [{
            CloseVariadicFieldRule(): Zero,
            XtoY: log(0.9), YsubtoNone: log(0.1)
        }]
        token0 = [{}]
        rule1 = [{
            CloseVariadicFieldRule(): Zero,
            XtoY: Zero, YsubtoNone: log(1.0)
        }]
        token1 = [{}]
        synthesizer = MockBeamSearchSynthesizer(2, is_subtype,
                                                [rule0, rule1],
                                                [token0, token1])
        candidates = []
        progress = []
        for c, p in synthesizer.synthesize("test"):
            candidates.extend(c)
            progress.append(p)
        """
        [] -> [XtoY] -> [XtoY, YsubtoNone] (Complete)
           -> [YsubtoNone] (Complete)
        """
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
        self.assertEqual([[0], [1]], synthesizer.arguments)

    def test_variadic_fields_generation(self):
        XtoY = ExpandTreeRule(X, [("value", Y_list)])
        YsubtoNone = ExpandTreeRule(Ysub, [])

        # Prepare mock probabilities
        rule0 = [{
            CloseVariadicFieldRule(): Zero, XtoY: log(0.9),
            YsubtoNone: log(0.1)
        }]
        token0 = [{}]
        rule1 = [{
            CloseVariadicFieldRule(): log(0.1), XtoY: Zero,
            YsubtoNone: log(0.9)
        }]
        token1 = [{}]
        rule2 = [{
            CloseVariadicFieldRule(): log(0.9), XtoY: Zero,
            YsubtoNone: log(0.1)
        }]
        token2 = [{}]

        synthesizer = MockBeamSearchSynthesizer(3, is_subtype,
                                                [rule0, rule1, rule2],
                                                [token0, token1, token2])
        candidates = []
        progress = []
        for c, p in synthesizer.synthesize("test"):
            candidates.extend(c)
            progress.append(p)
        """
        [] -> [XtoY] -> [XtoY, YsubtoNone] -> [XtoY, YsubtoNone, Close]
           -> [YsubtoNone] (Complete)
        """
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
        self.assertEqual([[0], [1], [2]], synthesizer.arguments)

    def test_retain_variadic_fields_False(self):
        XtoY = ExpandTreeRule(X, [("value", Y_list)])
        YtoYList = ExpandTreeRule(Y_list, [("0", Y)])
        YsubtoNone = ExpandTreeRule(Ysub, [])

        # Prepare mock probabilities
        rule0 = [{
            XtoY: log(0.9), YtoYList: Zero, YsubtoNone: log(0.1)
        }]
        token0 = [{}]
        rule1 = [{
            XtoY: log(0.1), YtoYList: log(0.9), YsubtoNone: Zero
        }]
        token1 = [{}]
        rule2 = [{
            XtoY: log(0.1), YtoYList: Zero, YsubtoNone: log(0.9)
        }]
        token2 = [{}]
        synthesizer = MockBeamSearchSynthesizer(
            3, is_subtype,
            [rule0, rule1, rule2],
            [token0, token1, token2],
            options=ActionOptions(False, True))
        candidates = []
        progress = []
        for c, p in synthesizer.synthesize("test"):
            candidates.extend(c)
            progress.append(p)
        """
        [] -> [XtoY] -> [XtoY, YtoYList] -> [..., YsubtoNone] (Complete)
           -> [YsubtoNone] (Complete)
        """
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
                     ApplyRule(YtoYList), False),
            progress[1][0]
        )
        self.assertSameProgress(
            Progress(4, 3, np.log(0.9) + np.log(0.9) + np.log(0.9),
                     ApplyRule(YsubtoNone), True),
            progress[2][0]
        )

        self.assertEqual(2, len(candidates))
        self.assertSameCandidate(
            Candidate(np.log(0.1), Node("Ysub", [])), candidates[0]
        )
        self.assertSameCandidate(
            Candidate(np.log(0.9) + np.log(0.9) + np.log(0.9),
                      Node("X", [Field("value", "Y", [Node("Ysub", [])])])),
            candidates[1]
        )
        self.assertEqual([[0], [1], [2]], synthesizer.arguments)

    def test_token_generation(self):
        XtoStr = ExpandTreeRule(X, [("value", Str)])

        # Prepare mock probabilities
        rule0 = [{
            CloseVariadicFieldRule(): Zero, XtoStr: log(1.0)
        }]
        token0 = [{}]
        rule1 = [{}]
        token1 = [{CloseNode(): log(0.1), "foo": log(0.9)}]
        rule2 = [{}]
        token2 = [{CloseNode(): log(1.0), "foo": Zero}]

        synthesizer = MockBeamSearchSynthesizer(2, is_subtype,
                                                [rule0, rule1, rule2],
                                                [token0, token1, token2])
        candidates = []
        progress = []
        for c, p in synthesizer.synthesize("test"):
            candidates.extend(c)
            progress.append(p)
        """
        [] -> [XtoStr] -> "foo" -> CloseNode (Complete)
                       -> CloseNode (Complete)
        """
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
        self.assertEqual([[0], [1], [2]], synthesizer.arguments)

    def test_split_non_terminal_False(self):
        XtoStr = ExpandTreeRule(X, [("value", Str)])

        # Prepare mock probabilities
        rule0 = [{CloseVariadicFieldRule(): Zero, XtoStr: log(1.0)}]
        token0 = [{}]
        rule1 = [{}]
        token1 = [{"foo": log(0.9), "test": log(0.1)}]

        synthesizer = MockBeamSearchSynthesizer(
            2, is_subtype,
            [rule0, rule1], [token0, token1],
            options=ActionOptions(True, False))
        candidates = []
        progress = []
        for c, p in synthesizer.synthesize("test"):
            candidates.extend(c)
            progress.append(p)
        """
        [] -> [XtoStr] -> "foo" -> (complete)
                       -> "test" -> (complete)
        """
        self.assertEqual(2, len(progress))
        self.assertSameProgress(
            Progress(1, 0, np.log(1.0), ApplyRule(XtoStr), False),
            progress[0][0]
        )
        self.assertSameProgress(
            Progress(2, 1, np.log(0.9), GenerateToken("foo"), True),
            progress[1][0]
        )
        self.assertSameProgress(
            Progress(3, 1, np.log(0.1), GenerateToken("test"), True),
            progress[1][1]
        )

        self.assertEqual(2, len(candidates))
        self.assertSameCandidate(
            Candidate(np.log(0.9),
                      Node("X", [Field("value", "Str", Leaf("Str", "foo"))])),
            candidates[0]
        )
        self.assertSameCandidate(
            Candidate(np.log(0.1),
                      Node("X", [Field("value", "Str", Leaf("Str", "test"))])),
            candidates[1]
        )
        self.assertEqual([[0], [1]], synthesizer.arguments)

    def test_not_generate_root_node(self):
        RoottoNone = ExpandTreeRule(NodeType(Root(), NodeConstraint.Node), [])
        XtoNone = ExpandTreeRule(X, [])

        # Prepare mock probabilities
        rule0 = [{
            CloseVariadicFieldRule(): Zero,
            XtoNone: log(0.1), RoottoNone: log(0.9)
        }]
        token0 = [{}]
        synthesizer = MockBeamSearchSynthesizer(1, is_subtype,
                                                [rule0], [token0])
        candidates = []
        progress = []
        for c, p in synthesizer.synthesize("test"):
            candidates.extend(c)
            progress.append(p)
        """
        [] -> [XtoNone] (Complete)
        """
        self.assertEqual(1, len(progress))
        self.assertSameProgress(
            Progress(1, 0, np.log(0.1), ApplyRule(XtoNone), True),
            progress[0][0]
        )
        self.assertEqual(1, len(candidates))
        self.assertSameCandidate(
            Candidate(np.log(0.1), Node("X", [])), candidates[0]
        )
        self.assertEqual([[0]], synthesizer.arguments)


if __name__ == "__main__":
    unittest.main()
