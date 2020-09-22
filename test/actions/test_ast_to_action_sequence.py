import unittest
from mlprogram.actions \
    import ApplyRule, ExpandTreeRule, NodeType, \
    NodeConstraint, GenerateToken, CloseVariadicFieldRule
from mlprogram.languages import ast
from mlprogram.languages.ast import Root
from mlprogram.actions import AstToActionSequence


class TestAstToSequence(unittest.TestCase):
    def test_leaf(self):
        f = AstToActionSequence()
        seq = f(ast.Leaf("str", "t0 t1"))
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
        seq = f(ast.Node("value", [ast.Field("name", "str",
                                             [ast.Leaf("str", "t0"),
                                              ast.Leaf("str", "t1")])]))
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
        a = ast.Node(
            "def",
            [ast.Field("name", "literal", ast.Leaf("str", "foo"))])
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
        a = ast.Node("list", [ast.Field("elems", "literal", [
            ast.Node("str", []), ast.Node("str", [])])])
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
