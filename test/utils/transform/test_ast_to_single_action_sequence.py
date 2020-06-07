import unittest
from mlprogram.action \
    import ActionOptions, ApplyRule, ExpandTreeRule, NodeType, \
    NodeConstraint, GenerateToken, CloseNode, CloseVariadicFieldRule
from mlprogram import ast
from mlprogram.ast import Root
from mlprogram.utils.transform import AstToSingleActionSequence


def tokenize(value: str):
    return value.split(" ")


class TestAstToActionSequence(unittest.TestCase):
    def test_leaf(self):
        f = AstToSingleActionSequence(ActionOptions(True, True),
                                      tokenize=tokenize)
        seq = f(ast.Leaf("str", "t0 t1"))
        self.assertEqual(
            [ApplyRule(ExpandTreeRule(NodeType(Root(), NodeConstraint.Node),
                                      [("root", NodeType(Root(),
                                                         NodeConstraint.Token))
                                       ])),
                GenerateToken("t0"), GenerateToken(
                "t1"), GenerateToken(CloseNode())],
            seq.action_sequence
        )
        self.assertEqual(ActionOptions(True, True), seq._options)

        f = AstToSingleActionSequence(ActionOptions(True, False))
        seq = f(ast.Leaf("str", "t0 t1"))
        self.assertEqual(
            [ApplyRule(ExpandTreeRule(NodeType(Root(), NodeConstraint.Node),
                                      [("root", NodeType(Root(),
                                                         NodeConstraint.Token))
                                       ])),
             GenerateToken("t0 t1")],
            seq.action_sequence
        )
        self.assertEqual(ActionOptions(True, False), seq._options)

    def test_node(self):
        a = ast.Node(
            "def",
            [ast.Field("name", "literal", ast.Leaf("str", "foo"))])
        f = AstToSingleActionSequence(ActionOptions(True, True),
                                      tokenize=tokenize)
        seq = f(a)
        self.assertEqual(
            [ApplyRule(ExpandTreeRule(NodeType(Root(), NodeConstraint.Node),
                                      [("root", NodeType(Root(),
                                                         NodeConstraint.Node))
                                       ])),
             ApplyRule(ExpandTreeRule(
                 NodeType("def", NodeConstraint.Node),
                 [("name",
                   NodeType("literal", NodeConstraint.Token))])),
             GenerateToken("foo"),
             GenerateToken(CloseNode())],
            seq.action_sequence
        )
        self.assertEqual(ActionOptions(True, True), seq._options)

    def test_node_with_variadic_fields(self):
        a = ast.Node("list", [ast.Field("elems", "literal", [
                     ast.Node("str", []), ast.Node("str", [])])])
        f = AstToSingleActionSequence(ActionOptions(True, True),
                                      tokenize=tokenize)
        seq = f(a)
        self.assertEqual(
            [ApplyRule(ExpandTreeRule(NodeType(Root(), NodeConstraint.Node),
                                      [("root", NodeType(Root(),
                                                         NodeConstraint.Node))
                                       ])),
             ApplyRule(ExpandTreeRule(
                 NodeType("list", NodeConstraint.Node),
                 [("elems",
                   NodeType("literal",
                            NodeConstraint.Variadic))])),
             ApplyRule(ExpandTreeRule(
                 NodeType("str", NodeConstraint.Node),
                 [])),
             ApplyRule(ExpandTreeRule(
                 NodeType("str", NodeConstraint.Node),
                 [])),
             ApplyRule(CloseVariadicFieldRule())],
            seq.action_sequence
        )
        self.assertEqual(seq._options, ActionOptions(True, True))

        f = AstToSingleActionSequence(ActionOptions(False, True),
                                      tokenize=tokenize)
        seq = f(a)
        self.assertEqual(
            [ApplyRule(ExpandTreeRule(NodeType(Root(), NodeConstraint.Node),
                                      [("root", NodeType(Root(),
                                                         NodeConstraint.Node))
                                       ])),
             ApplyRule(ExpandTreeRule(
                 NodeType("list", NodeConstraint.Node),
                 [("elems",
                   NodeType("literal",
                            NodeConstraint.Variadic))])),
             ApplyRule(ExpandTreeRule(
                 NodeType("literal", NodeConstraint.Variadic),
                 [("0", NodeType("literal", NodeConstraint.Node)),
                  ("1", NodeType("literal", NodeConstraint.Node))])),
             ApplyRule(ExpandTreeRule(
                 NodeType("str", NodeConstraint.Node),
                 [])),
             ApplyRule(ExpandTreeRule(
                 NodeType("str", NodeConstraint.Node),
                 []))],
            seq.action_sequence
        )
        self.assertEqual(ActionOptions(False, True), seq._options)


if __name__ == "__main__":
    unittest.main()
