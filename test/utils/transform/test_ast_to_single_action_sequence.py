import unittest
from mlprogram.action.action \
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
            seq.sequence
        )
        self.assertEqual(ActionOptions(True, True), seq.options)

        f = AstToSingleActionSequence(ActionOptions(True, False))
        seq = f(ast.Leaf("str", "t0 t1"))
        self.assertEqual(
            [ApplyRule(ExpandTreeRule(NodeType(Root(), NodeConstraint.Node),
                                      [("root", NodeType(Root(),
                                                         NodeConstraint.Token))
                                       ])),
             GenerateToken("t0 t1")],
            seq.sequence
        )
        self.assertEqual(ActionOptions(True, False), seq.options)

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
            seq.sequence
        )
        self.assertEqual(ActionOptions(True, True), seq.options)

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
            seq.sequence
        )
        self.assertEqual(seq.options, ActionOptions(True, True))

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
            seq.sequence
        )
        self.assertEqual(ActionOptions(False, True), seq.options)


if __name__ == "__main__":
    unittest.main()
