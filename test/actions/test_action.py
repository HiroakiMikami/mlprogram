import unittest

from mlprogram.actions \
    import ExpandTreeRule, GenerateToken, ApplyRule, NodeType, \
    NodeConstraint, CloseVariadicFieldRule


class TestNodeType(unittest.TestCase):
    def test_str(self):
        self.assertEqual("type",
                         str(NodeType("type", NodeConstraint.Node, False)))
        self.assertEqual("type*",
                         str(NodeType("type", NodeConstraint.Node, True)))
        self.assertEqual("type(token)",
                         str(NodeType("type", NodeConstraint.Token, False)))

    def test_eq(self):
        self.assertEqual(NodeType("foo", NodeConstraint.Node, False),
                         NodeType("foo", NodeConstraint.Node, False))
        self.assertNotEqual(NodeType("foo", NodeConstraint.Node, False),
                            NodeType("foo", NodeConstraint.Node, True))
        self.assertNotEqual(0, NodeType("foo", NodeConstraint.Node, False))


class TestRule(unittest.TestCase):
    def test_str(self):
        t0 = NodeType("t0", NodeConstraint.Node, False)
        t1 = NodeType("t1", NodeConstraint.Node, False)
        t2 = NodeType("t2", NodeConstraint.Node, True)
        self.assertEqual("t0 -> [elem0: t1, elem1: t2*]",
                         str(ExpandTreeRule(t0, [("elem0", t1),
                                                 ("elem1", t2)])))
        self.assertEqual("<close variadic field>", str(
            CloseVariadicFieldRule()))

    def test_eq(self):
        self.assertEqual(
            ExpandTreeRule(NodeType("foo", NodeConstraint.Node, False), [
                ("f0", NodeType("bar", NodeConstraint.Node, False))]),
            ExpandTreeRule(
                NodeType("foo", NodeConstraint.Node, False),
                [("f0", NodeType("bar", NodeConstraint.Node, False))]))
        self.assertEqual(
            GenerateToken("foo"), GenerateToken("foo"))
        self.assertNotEqual(
            ExpandTreeRule(NodeType("foo", NodeConstraint.Node, False), [
                ("f0", NodeType("bar", NodeConstraint.Node, False))]),
            ExpandTreeRule(NodeType("foo", NodeConstraint.Node, False), []))
        self.assertNotEqual(
            GenerateToken("foo"), GenerateToken("bar"))
        self.assertNotEqual(
            ExpandTreeRule(NodeType("foo", NodeConstraint.Node, False), [
                ("f0", NodeType("bar", NodeConstraint.Node, False))]),
            GenerateToken("foo"))
        self.assertNotEqual(
            0,
            ExpandTreeRule(
                NodeType("foo", NodeConstraint.Node, False),
                [("f0", NodeType("bar", NodeConstraint.Node, False))]))


class TestAction(unittest.TestCase):
    def test_str(self):
        t0 = NodeType("t0", NodeConstraint.Node, False)
        t1 = NodeType("t1", NodeConstraint.Node, False)
        t2 = NodeType("t2", NodeConstraint.Node, True)
        self.assertEqual("Apply (t0 -> [elem0: t1, elem1: t2*])",
                         str(ApplyRule(
                             ExpandTreeRule(t0,
                                            [("elem0", t1),
                                             ("elem1", t2)]))))

        self.assertEqual("Generate foo", str(GenerateToken("foo")))


if __name__ == "__main__":
    unittest.main()
