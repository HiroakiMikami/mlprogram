import unittest
from mlprogram.actions \
    import ApplyRule, ExpandTreeRule, NodeType, \
    NodeConstraint, GenerateToken, CloseVariadicFieldRule
from mlprogram.languages import Root
from mlprogram.languages import Node
from mlprogram.languages import Field
from mlprogram.languages import Leaf
from mlprogram.actions import AstToActionSequence


class TestAstToSequence(unittest.TestCase):
    def test_leaf(self):
        f = AstToActionSequence()
        seq = f(Leaf("str", "t0 t1"))
        self.assertEqual(
            [ApplyRule(ExpandTreeRule(
                NodeType(None, NodeConstraint.Node, False),
                [("root", NodeType(Root(),
                                   NodeConstraint.Token, False))
                 ])),
                GenerateToken("t0 t1")],
            seq.action_sequence
        )

        f = AstToActionSequence()
        seq = f(Node("value", [Field("name", "str",
                                             [Leaf("str", "t0"),
                                              Leaf("str", "t1")])]))
        self.assertEqual(
            [ApplyRule(ExpandTreeRule(
                NodeType(None, NodeConstraint.Node, False),
                [("root", NodeType(Root(),
                                   NodeConstraint.Node, False))
                 ])),
             ApplyRule(ExpandTreeRule(
                 NodeType("value", NodeConstraint.Node, False),
                 [("name", NodeType("str", NodeConstraint.Token, True))]
             )),
             GenerateToken("t0"),
             GenerateToken("t1"),
             ApplyRule(CloseVariadicFieldRule())],
            seq.action_sequence
        )

    def test_node(self):
        a = Node(
            "def",
            [Field("name", "literal", Leaf("str", "foo"))])
        f = AstToActionSequence()
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
                   NodeType("literal", NodeConstraint.Token, False))])),
             GenerateToken("foo")],
            seq.action_sequence
        )

    def test_node_with_variadic_fields(self):
        a = Node("list", [Field("elems", "literal", [
            Node("str", []), Node("str", [])])])
        f = AstToActionSequence()
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


if __name__ == "__main__":
    unittest.main()
