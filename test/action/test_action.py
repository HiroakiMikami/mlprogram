import unittest

from mlprogram.action.action \
    import ExpandTreeRule, GenerateToken, ApplyRule, NodeType, \
    NodeConstraint, CloseNode, CloseVariadicFieldRule, \
    ActionOptions, ast_to_action_sequence
from mlprogram.action import ast
from mlprogram.action.ast import Root


class TestNodeType(unittest.TestCase):
    def test_str(self):
        self.assertEqual("type", str(NodeType("type", NodeConstraint.Node)))
        self.assertEqual("type*",
                         str(NodeType("type", NodeConstraint.Variadic)))
        self.assertEqual("type(token)",
                         str(NodeType("type", NodeConstraint.Token)))

    def test_eq(self):
        self.assertEqual(NodeType("foo", NodeConstraint.Node),
                         NodeType("foo", NodeConstraint.Node))
        self.assertNotEqual(NodeType("foo", NodeConstraint.Node),
                            NodeType("foo", NodeConstraint.Variadic))
        self.assertNotEqual(0, NodeType("foo", NodeConstraint.Node))


class TestRule(unittest.TestCase):
    def test_str(self):
        t0 = NodeType("t0", NodeConstraint.Node)
        t1 = NodeType("t1", NodeConstraint.Node)
        t2 = NodeType("t2", NodeConstraint.Variadic)
        self.assertEqual("t0 -> [elem0: t1, elem1: t2*]",
                         str(ExpandTreeRule(t0, [("elem0", t1),
                                                 ("elem1", t2)])))
        self.assertEqual("<close variadic field>", str(
            CloseVariadicFieldRule()))

    def test_eq(self):
        self.assertEqual(
            ExpandTreeRule(NodeType("foo", NodeConstraint.Node), [
                ("f0", NodeType("bar", NodeConstraint.Node))]),
            ExpandTreeRule(NodeType("foo", NodeConstraint.Node),
                           [("f0", NodeType("bar", NodeConstraint.Node))]))
        self.assertEqual(
            GenerateToken("foo"), GenerateToken("foo"))
        self.assertNotEqual(
            ExpandTreeRule(NodeType("foo", NodeConstraint.Node), [
                ("f0", NodeType("bar", NodeConstraint.Node))]),
            ExpandTreeRule(NodeType("foo", NodeConstraint.Node), []))
        self.assertNotEqual(
            GenerateToken("foo"), GenerateToken("bar"))
        self.assertNotEqual(
            ExpandTreeRule(NodeType("foo", NodeConstraint.Node), [
                ("f0", NodeType("bar", NodeConstraint.Node))]),
            GenerateToken("foo"))
        self.assertNotEqual(
            0,
            ExpandTreeRule(NodeType("foo", NodeConstraint.Node),
                           [("f0", NodeType("bar", NodeConstraint.Node))]))


class TestAction(unittest.TestCase):
    def test_str(self):
        t0 = NodeType("t0", NodeConstraint.Node)
        t1 = NodeType("t1", NodeConstraint.Node)
        t2 = NodeType("t2", NodeConstraint.Variadic)
        self.assertEqual("Apply (t0 -> [elem0: t1, elem1: t2*])",
                         str(ApplyRule(
                             ExpandTreeRule(t0,
                                            [("elem0", t1),
                                             ("elem1", t2)]))))

        self.assertEqual("Generate foo", str(GenerateToken("foo")))
        self.assertEqual("Generate <CLOSE_NODE>", str(
            GenerateToken(CloseNode())))


def tokenize(value: str):
    return value.split(" ")


class TestAstToActionSequence(unittest.TestCase):
    def test_leaf(self):
        seq = ast_to_action_sequence(ast.Leaf("str", "t0 t1"),
                                     ActionOptions(True, True),
                                     tokenizer=tokenize)
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
        seq = ast_to_action_sequence(ast.Leaf("str", "t0 t1"),
                                     ActionOptions(True, False))
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
        seq = ast_to_action_sequence(a, ActionOptions(True, True),
                                     tokenizer=tokenize)
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
        seq = ast_to_action_sequence(a, ActionOptions(True, True),
                                     tokenizer=tokenize)
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
        seq = ast_to_action_sequence(a, options=ActionOptions(False, True))
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
