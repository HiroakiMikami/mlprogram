import unittest

from nl2code.language.action import NodeType, NodeConstraint
from nl2code.language.action import ExpandTreeRule, GenerateToken, ApplyRule
from nl2code.language.action import CloseNode, CloseVariadicFieldRule


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


if __name__ == "__main__":
    unittest.main()
