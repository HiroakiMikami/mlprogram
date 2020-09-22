import unittest

from mlprogram.actions.action_sequence import Parent
from mlprogram.actions import ActionSequence, InvalidActionException
from mlprogram.actions \
    import ExpandTreeRule, NodeType, NodeConstraint, ApplyRule, \
    GenerateToken, CloseVariadicFieldRule
from mlprogram.languages.ast import Node, Leaf, Field, Root


class Testaction_sequence(unittest.TestCase):
    def test_eval_root(self):
        action_sequence = ActionSequence()
        self.assertEqual(None, action_sequence.head)
        with self.assertRaises(InvalidActionException):
            action_sequence = ActionSequence()
            action_sequence.eval(GenerateToken(""))
        with self.assertRaises(InvalidActionException):
            action_sequence = ActionSequence()
            action_sequence.eval(ApplyRule(CloseVariadicFieldRule()))

        action_sequence = ActionSequence()
        rule = ExpandTreeRule(NodeType("def", NodeConstraint.Node, False),
                              [("name",
                                NodeType("value", NodeConstraint.Node, False)),
                               ("value",
                                NodeType("args", NodeConstraint.Node, True))])
        action_sequence.eval(ApplyRule(rule))
        self.assertEqual(0, action_sequence.head.action)
        self.assertEqual(0, action_sequence.head.field)
        self.assertEqual([ApplyRule(rule)], action_sequence.action_sequence)
        self.assertEqual(None, action_sequence.parent(0))
        self.assertEqual([[], []], action_sequence._tree.children[0])

    def test_generate_variadic_token(self):
        action_sequence = ActionSequence()
        rule = ExpandTreeRule(
            NodeType("def", NodeConstraint.Node, False),
            [("name",
              NodeType("value", NodeConstraint.Token, True)),
             ("value",
              NodeType("args", NodeConstraint.Node, True))])
        action_sequence.eval(ApplyRule(rule))
        action_sequence.eval(GenerateToken("foo"))
        self.assertEqual(0, action_sequence.head.action)
        self.assertEqual(0, action_sequence.head.field)
        self.assertEqual([1], action_sequence._tree.children[0][0])
        self.assertEqual([ApplyRule(rule), GenerateToken("foo")],
                         action_sequence.action_sequence)
        self.assertEqual(Parent(0, 0), action_sequence.parent(1))
        self.assertEqual([], action_sequence._tree.children[1])

        action_sequence.eval(GenerateToken("bar"))
        self.assertEqual(0, action_sequence.head.action)
        self.assertEqual(0, action_sequence.head.field)
        self.assertEqual([1, 2], action_sequence._tree.children[0][0])
        self.assertEqual([ApplyRule(rule),
                          GenerateToken("foo"), GenerateToken("bar")],
                         action_sequence.action_sequence)

        action_sequence.eval(ApplyRule(CloseVariadicFieldRule()))
        self.assertEqual(0, action_sequence.head.action)
        self.assertEqual(1, action_sequence.head.field)
        self.assertEqual([1, 2, 3], action_sequence._tree.children[0][0])
        self.assertEqual([ApplyRule(rule),
                          GenerateToken("foo"), GenerateToken("bar"),
                          ApplyRule(CloseVariadicFieldRule())],
                         action_sequence.action_sequence)

        with self.assertRaises(InvalidActionException):
            action_sequence.eval(GenerateToken("foo"))

    def test_generate_token(self):
        action_sequence = ActionSequence()
        rule = ExpandTreeRule(
            NodeType("def", NodeConstraint.Node, False),
            [("name",
              NodeType("value", NodeConstraint.Token, False)),
             ("value",
              NodeType("args", NodeConstraint.Node, True))])
        action_sequence.eval(ApplyRule(rule))
        action_sequence.eval(GenerateToken("foo"))
        self.assertEqual(0, action_sequence.head.action)
        self.assertEqual(1, action_sequence.head.field)
        self.assertEqual([1], action_sequence._tree.children[0][0])
        self.assertEqual([ApplyRule(rule), GenerateToken("foo")],
                         action_sequence.action_sequence)
        self.assertEqual(Parent(0, 0), action_sequence.parent(1))
        self.assertEqual([], action_sequence._tree.children[1])

        with self.assertRaises(InvalidActionException):
            action_sequence.eval(GenerateToken("bar"))

        action_sequence = ActionSequence()
        action_sequence.eval(ApplyRule(rule))
        with self.assertRaises(InvalidActionException):
            action_sequence.eval(ApplyRule(CloseVariadicFieldRule()))

    def test_variadic_field(self):
        action_sequence = ActionSequence()
        rule = ExpandTreeRule(NodeType("expr", NodeConstraint.Node, False),
                              [("elems",
                                NodeType("value", NodeConstraint.Node, True))])
        rule0 = ExpandTreeRule(NodeType("value", NodeConstraint.Node, False),
                               [])
        action_sequence.eval(ApplyRule(rule))
        action_sequence.eval(ApplyRule(rule0))
        self.assertEqual(0, action_sequence.head.action)
        self.assertEqual(0, action_sequence.head.field)
        self.assertEqual([1], action_sequence._tree.children[0][0])
        self.assertEqual([ApplyRule(rule), ApplyRule(rule0)],
                         action_sequence.action_sequence)
        self.assertEqual(Parent(0, 0), action_sequence.parent(1))
        self.assertEqual([], action_sequence._tree.children[1])

        action_sequence.eval(ApplyRule(rule0))
        self.assertEqual(0, action_sequence.head.action)
        self.assertEqual(0, action_sequence.head.field)
        self.assertEqual([1, 2], action_sequence._tree.children[0][0])
        self.assertEqual([ApplyRule(rule), ApplyRule(rule0), ApplyRule(rule0)],
                         action_sequence.action_sequence)

        action_sequence.eval(ApplyRule(CloseVariadicFieldRule()))
        self.assertEqual(None, action_sequence.head)

        action_sequence = ActionSequence()
        rule1 = ExpandTreeRule(
            NodeType("expr", NodeConstraint.Node, False),
            [("elems",
              NodeType("value", NodeConstraint.Node, True)),
             ("name",
              NodeType("value", NodeConstraint.Node, False))])
        rule0 = ExpandTreeRule(NodeType("value", NodeConstraint.Node, False),
                               [])
        action_sequence.eval(ApplyRule(rule1))
        action_sequence.eval(ApplyRule(rule0))
        action_sequence.eval(ApplyRule(CloseVariadicFieldRule))
        self.assertEqual(0, action_sequence.head.action)
        self.assertEqual(1, action_sequence.head.field)

    def test_invalid_close_variadic_field_rule(self):
        rule = ExpandTreeRule(NodeType("expr", NodeConstraint.Node, False),
                              [("elems",
                                NodeType("value", NodeConstraint.Node, False))]
                              )

        action_sequence = ActionSequence()
        action_sequence.eval(ApplyRule(rule))
        with self.assertRaises(InvalidActionException):
            action_sequence.eval(ApplyRule(CloseVariadicFieldRule()))

    def test_generate(self):
        funcdef = ExpandTreeRule(
            NodeType("def", NodeConstraint.Node, False),
            [("name",
              NodeType("value", NodeConstraint.Token, True)),
             ("body",
              NodeType("expr", NodeConstraint.Node, True))])
        expr = ExpandTreeRule(
            NodeType("expr", NodeConstraint.Node, False),
            [("op", NodeType("value", NodeConstraint.Token, False)),
             ("arg0",
              NodeType("value", NodeConstraint.Token, False)),
             ("arg1",
              NodeType("value", NodeConstraint.Token, False))])

        action_sequence = ActionSequence()
        action_sequence.eval(ApplyRule(funcdef))
        action_sequence.eval(GenerateToken("f"))
        action_sequence.eval(GenerateToken("_"))
        action_sequence.eval(GenerateToken("0"))
        action_sequence.eval(ApplyRule(CloseVariadicFieldRule()))
        action_sequence.eval(ApplyRule(expr))
        action_sequence.eval(GenerateToken("+"))
        action_sequence.eval(GenerateToken("1"))
        action_sequence.eval(GenerateToken("2"))
        action_sequence.eval(ApplyRule(CloseVariadicFieldRule()))
        self.assertEqual(None, action_sequence.head)
        self.assertEqual(
            Node("def",
                 [Field("name", "value", [Leaf("value", "f"),
                                          Leaf("value", "_"),
                                          Leaf("value", "0")]),
                  Field("body", "expr", [
                      Node("expr",
                           [
                               Field("op", "value", Leaf("value", "+")),
                               Field("arg0", "value", Leaf("value", "1")),
                               Field("arg1", "value", Leaf("value", "2"))
                           ])
                  ])]),
            action_sequence.generate()
        )

    def test_generate_ignore_root_type(self):
        action_sequence = ActionSequence()
        action_sequence.eval(ApplyRule(ExpandTreeRule(
            NodeType(Root(), NodeConstraint.Node, False),
            [("root", NodeType(Root(), NodeConstraint.Node, False))])))
        action_sequence.eval(ApplyRule(ExpandTreeRule(
            NodeType("op", NodeConstraint.Node, False), []
        )))
        self.assertEqual(Node("op", []), action_sequence.generate())

    def test_clone(self):
        action_sequence = ActionSequence()
        rule = ExpandTreeRule(NodeType("expr", NodeConstraint.Node, False),
                              [("elems",
                                NodeType("expr", NodeConstraint.Node, True))])
        action_sequence.eval(ApplyRule(rule))

        action_sequence2 = action_sequence.clone()
        self.assertEqual(action_sequence.generate(),
                         action_sequence2.generate())

        action_sequence2.eval(ApplyRule(rule))
        self.assertNotEqual(action_sequence._tree.children,
                            action_sequence2._tree.children)
        self.assertNotEqual(action_sequence._tree.parent,
                            action_sequence2._tree.parent)
        self.assertNotEqual(action_sequence.action_sequence,
                            action_sequence2.action_sequence)
        self.assertNotEqual(action_sequence._head_action_index,
                            action_sequence2._head_action_index)
        self.assertNotEqual(action_sequence._head_children_index,
                            action_sequence2._head_children_index)
        self.assertNotEqual(action_sequence.generate(),
                            action_sequence2.generate())


if __name__ == "__main__":
    unittest.main()
