import unittest
from mlprogram.actions \
    import ActionOptions, ApplyRule, ExpandTreeRule, NodeType, \
    NodeConstraint, GenerateToken, CloseVariadicFieldRule
from mlprogram import asts
from mlprogram.asts import Root
from mlprogram.utils.transform import AstToSingleActionSequence


def tokenize(value: str):
    return value.split(" ")


class TestAstToActionSequence(unittest.TestCase):
    def test_leaf(self):
        f = AstToSingleActionSequence(ActionOptions(True, True),
                                      tokenize=tokenize)
        seq = f(asts.Leaf("str", "t0 t1"))
        self.assertEqual(
            [ApplyRule(ExpandTreeRule(
                NodeType(None, NodeConstraint.Node, False),
                [("root", NodeType(Root(),
                                   NodeConstraint.Token, True))
                 ])),
                GenerateToken("t0"), GenerateToken("t1"),
                ApplyRule(CloseVariadicFieldRule())],
            seq.action_sequence
        )
        self.assertEqual(ActionOptions(True, True), seq._options)

        f = AstToSingleActionSequence(ActionOptions(True, False))
        seq = f(asts.Leaf("str", "t0 t1"))
        self.assertEqual(
            [ApplyRule(ExpandTreeRule(
                NodeType(None, NodeConstraint.Node, False),
                [("root", NodeType(Root(),
                                   NodeConstraint.Token, False))
                 ])),
             GenerateToken("t0 t1")],
            seq.action_sequence
        )
        self.assertEqual(ActionOptions(True, False), seq._options)

    def test_node(self):
        a = asts.Node(
            "def",
            [asts.Field("name", "literal", asts.Leaf("str", "foo"))])
        f = AstToSingleActionSequence(ActionOptions(True, True),
                                      tokenize=tokenize)
        seq = f(a)
        self.assertEqual(
            [ApplyRule(ExpandTreeRule(
                NodeType(None, NodeConstraint.Node, False),
                [("root", NodeType(Root(),
                                   NodeConstraint.Node, False))
                 ])),
             ApplyRule(ExpandTreeRule(
                 NodeType("def", NodeConstraint.Node, False),
                 [("name",
                   NodeType("literal", NodeConstraint.Token, True))])),
             GenerateToken("foo"),
             ApplyRule(CloseVariadicFieldRule())],
            seq.action_sequence
        )
        self.assertEqual(ActionOptions(True, True), seq._options)

    def test_node_with_variadic_fields(self):
        a = asts.Node("list", [asts.Field("elems", "literal", [
            asts.Node("str", []), asts.Node("str", [])])])
        f = AstToSingleActionSequence(ActionOptions(True, True),
                                      tokenize=tokenize)
        seq = f(a)
        self.assertEqual(
            [ApplyRule(ExpandTreeRule(
                NodeType(None, NodeConstraint.Node, False),
                [("root", NodeType(Root(),
                                   NodeConstraint.Node, False))
                 ])),
             ApplyRule(ExpandTreeRule(
                 NodeType("list", NodeConstraint.Node, False),
                 [("elems",
                   NodeType("literal",
                            NodeConstraint.Node, True))])),
             ApplyRule(ExpandTreeRule(
                 NodeType("str", NodeConstraint.Node, False),
                 [])),
             ApplyRule(ExpandTreeRule(
                 NodeType("str", NodeConstraint.Node, False),
                 [])),
             ApplyRule(CloseVariadicFieldRule())],
            seq.action_sequence
        )
        self.assertEqual(seq._options, ActionOptions(True, True))

        f = AstToSingleActionSequence(ActionOptions(False, True),
                                      tokenize=tokenize)
        seq = f(a)
        self.assertEqual(
            [ApplyRule(ExpandTreeRule(
                NodeType(None, NodeConstraint.Node, False),
                [("root", NodeType(Root(),
                                   NodeConstraint.Node, False))
                 ])),
             ApplyRule(ExpandTreeRule(
                 NodeType("list", NodeConstraint.Node, False),
                 [("elems",
                   NodeType("literal",
                            NodeConstraint.Node, True))])),
             ApplyRule(ExpandTreeRule(
                 NodeType("literal", NodeConstraint.Node, True),
                 [("0", NodeType("literal", NodeConstraint.Node, False)),
                  ("1", NodeType("literal", NodeConstraint.Node, False))])),
             ApplyRule(ExpandTreeRule(
                 NodeType("str", NodeConstraint.Node, False),
                 [])),
             ApplyRule(ExpandTreeRule(
                 NodeType("str", NodeConstraint.Node, False),
                 []))],
            seq.action_sequence
        )
        self.assertEqual(ActionOptions(False, True), seq._options)


if __name__ == "__main__":
    unittest.main()
