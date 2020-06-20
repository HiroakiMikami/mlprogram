import unittest

from mlprogram.actions.action_sequence import Parent
from mlprogram.actions import ActionSequence, InvalidActionException
from mlprogram.actions \
    import ExpandTreeRule, NodeType, NodeConstraint, ApplyRule, \
    GenerateToken, CloseVariadicFieldRule, CloseNode, ActionOptions
from mlprogram.asts import Node, Leaf, Field, Root


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
        rule = ExpandTreeRule(NodeType("def", NodeConstraint.Node),
                              [("name",
                                NodeType("value", NodeConstraint.Node)),
                               ("value",
                                NodeType("args", NodeConstraint.Variadic))])
        action_sequence.eval(ApplyRule(rule))
        self.assertEqual(0, action_sequence.head.action)
        self.assertEqual(0, action_sequence.head.field)
        self.assertEqual([ApplyRule(rule)], action_sequence.action_sequence)
        self.assertEqual(None, action_sequence.parent(0))
        self.assertEqual([[], []], action_sequence._tree.children[0])

    def test_generate_token(self):
        action_sequence = ActionSequence()
        rule = ExpandTreeRule(NodeType("def", NodeConstraint.Node),
                              [("name",
                                NodeType("value", NodeConstraint.Token)),
                               ("value",
                                NodeType("args", NodeConstraint.Variadic))])
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

        action_sequence.eval(GenerateToken(CloseNode()))
        self.assertEqual(0, action_sequence.head.action)
        self.assertEqual(1, action_sequence.head.field)
        self.assertEqual([1, 2, 3], action_sequence._tree.children[0][0])
        self.assertEqual([ApplyRule(rule),
                          GenerateToken("foo"), GenerateToken("bar"),
                          GenerateToken(CloseNode())],
                         action_sequence.action_sequence)

        with self.assertRaises(InvalidActionException):
            action_sequence.eval(GenerateToken("foo"))

    def test_generate_token_with_split_non_terminal_False(self):
        action_sequence = ActionSequence(ActionOptions(True, False))
        rule = ExpandTreeRule(NodeType("def", NodeConstraint.Node),
                              [("name",
                                NodeType("value", NodeConstraint.Token)),
                               ("value",
                                NodeType("args", NodeConstraint.Variadic))])
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

        action_sequence = ActionSequence(ActionOptions(True, False))
        action_sequence.eval(ApplyRule(rule))
        with self.assertRaises(InvalidActionException):
            action_sequence.eval(GenerateToken(CloseNode()))

    def test_variadic_field(self):
        action_sequence = ActionSequence()
        rule = ExpandTreeRule(NodeType("expr", NodeConstraint.Node),
                              [("elems",
                                NodeType("value", NodeConstraint.Variadic))])
        rule0 = ExpandTreeRule(NodeType("value", NodeConstraint.Node),
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
        rule1 = ExpandTreeRule(NodeType("expr", NodeConstraint.Node),
                               [("elems",
                                 NodeType("value", NodeConstraint.Variadic)),
                                ("name",
                                 NodeType("value", NodeConstraint.Node))])
        rule0 = ExpandTreeRule(NodeType("value", NodeConstraint.Node),
                               [])
        action_sequence.eval(ApplyRule(rule1))
        action_sequence.eval(ApplyRule(rule0))
        action_sequence.eval(ApplyRule(CloseVariadicFieldRule))
        self.assertEqual(0, action_sequence.head.action)
        self.assertEqual(1, action_sequence.head.field)

    def test_variadic_field_retain_variadic_fields_False(self):
        action_sequence = ActionSequence(ActionOptions(False, True))
        rule = ExpandTreeRule(NodeType("expr", NodeConstraint.Node),
                              [("elems",
                                NodeType("value", NodeConstraint.Variadic))])
        rule0 = ExpandTreeRule(NodeType("value", NodeConstraint.Variadic),
                               [("0", NodeType("value", NodeConstraint.Node)),
                                ("1", NodeType("value", NodeConstraint.Node))])
        rule1 = ExpandTreeRule(NodeType("value", NodeConstraint.Node),
                               [])
        action_sequence.eval(ApplyRule(rule))
        self.assertEqual(0, action_sequence.head.action)
        self.assertEqual(0, action_sequence.head.field)
        self.assertEqual([], action_sequence._tree.children[0][0])
        self.assertEqual([ApplyRule(rule)],
                         action_sequence.action_sequence)
        action_sequence.eval(ApplyRule(rule0))
        self.assertEqual(1, action_sequence.head.action)
        self.assertEqual(0, action_sequence.head.field)
        self.assertEqual([1], action_sequence._tree.children[0][0])
        self.assertEqual(Parent(0, 0), action_sequence.parent(1))
        self.assertEqual([ApplyRule(rule), ApplyRule(rule0)],
                         action_sequence.action_sequence)
        action_sequence.eval(ApplyRule(rule1))
        self.assertEqual(1, action_sequence.head.action)
        self.assertEqual(1, action_sequence.head.field)
        self.assertEqual([2], action_sequence._tree.children[1][0])
        self.assertEqual([], action_sequence._tree.children[2])
        self.assertEqual(Parent(1, 0), action_sequence.parent(2))
        self.assertEqual([ApplyRule(rule), ApplyRule(rule0), ApplyRule(rule1)],
                         action_sequence.action_sequence)
        action_sequence.eval(ApplyRule(rule1))
        self.assertEqual(None, action_sequence.head)
        self.assertEqual([3], action_sequence._tree.children[1][1])
        self.assertEqual([], action_sequence._tree.children[3])
        self.assertEqual(Parent(1, 1), action_sequence.parent(3))
        self.assertEqual([ApplyRule(rule), ApplyRule(rule0), ApplyRule(rule1),
                          ApplyRule(rule1)],
                         action_sequence.action_sequence)

        action_sequence = ActionSequence(ActionOptions(False, True))
        rule = ExpandTreeRule(NodeType("expr", NodeConstraint.Node),
                              [("elems",
                                NodeType("value", NodeConstraint.Variadic)),
                               ("name",
                                NodeType("value", NodeConstraint.Node))])
        action_sequence.eval(ApplyRule(rule))
        action_sequence.eval(ApplyRule(rule0))
        action_sequence.eval(ApplyRule(rule1))
        action_sequence.eval(ApplyRule(rule1))
        self.assertEqual(0, action_sequence.head.action)
        self.assertEqual(1, action_sequence.head.field)

        action_sequence = ActionSequence(ActionOptions(False, True))
        with self.assertRaises(InvalidActionException):
            action_sequence.eval(ApplyRule(CloseVariadicFieldRule()))

    def test_generate(self):
        funcdef = ExpandTreeRule(NodeType("def", NodeConstraint.Node),
                                 [("name",
                                   NodeType("value", NodeConstraint.Token)),
                                  ("body",
                                   NodeType("expr", NodeConstraint.Variadic))])
        expr_expand = ExpandTreeRule(NodeType("expr", NodeConstraint.Variadic),
                                     [("0",
                                       NodeType("expr", NodeConstraint.Node))])
        expr = ExpandTreeRule(NodeType("expr", NodeConstraint.Node),
                              [("op", NodeType("value", NodeConstraint.Token)),
                               ("arg0",
                                NodeType("value", NodeConstraint.Token)),
                               ("arg1",
                                NodeType("value", NodeConstraint.Token))])

        action_sequence = ActionSequence()
        action_sequence.eval(ApplyRule(funcdef))
        action_sequence.eval(GenerateToken("f"))
        action_sequence.eval(GenerateToken("_"))
        action_sequence.eval(GenerateToken("0"))
        action_sequence.eval(GenerateToken(CloseNode()))
        action_sequence.eval(ApplyRule(expr))
        action_sequence.eval(GenerateToken("+"))
        action_sequence.eval(GenerateToken(CloseNode()))
        action_sequence.eval(GenerateToken("1"))
        action_sequence.eval(GenerateToken(CloseNode()))
        action_sequence.eval(GenerateToken("2"))
        action_sequence.eval(GenerateToken(CloseNode()))
        action_sequence.eval(ApplyRule(CloseVariadicFieldRule()))
        self.assertEqual(None, action_sequence.head)
        self.assertEqual(
            Node("def",
                 [Field("name", "value", Leaf("value", "f_0")),
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

        action_sequence = ActionSequence(ActionOptions(False, True))
        action_sequence.eval(ApplyRule(funcdef))
        action_sequence.eval(GenerateToken("f_0"))
        action_sequence.eval(GenerateToken(CloseNode()))
        action_sequence.eval(ApplyRule(expr_expand))
        action_sequence.eval(ApplyRule(expr))
        action_sequence.eval(GenerateToken("+"))
        action_sequence.eval(GenerateToken(CloseNode()))
        action_sequence.eval(GenerateToken("1"))
        action_sequence.eval(GenerateToken(CloseNode()))
        action_sequence.eval(GenerateToken("2"))
        action_sequence.eval(GenerateToken(CloseNode()))
        self.assertEqual(None, action_sequence.head)
        self.assertEqual(
            Node("def",
                 [Field("name", "value", Leaf("value", "f_0")),
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

        action_sequence = ActionSequence(ActionOptions(True, False))
        action_sequence.eval(ApplyRule(funcdef))
        action_sequence.eval(GenerateToken("f_0"))
        action_sequence.eval(ApplyRule(expr))
        action_sequence.eval(GenerateToken("+"))
        action_sequence.eval(GenerateToken("1"))
        action_sequence.eval(GenerateToken("2"))
        action_sequence.eval(ApplyRule(CloseVariadicFieldRule()))
        self.assertEqual(None, action_sequence.head)
        self.assertEqual(
            Node("def",
                 [Field("name", "value", Leaf("value", "f_0")),
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

        action_sequence = ActionSequence(ActionOptions(False, False))
        action_sequence.eval(ApplyRule(funcdef))
        action_sequence.eval(GenerateToken("f_0"))
        action_sequence.eval(ApplyRule(expr_expand))
        action_sequence.eval(ApplyRule(expr))
        action_sequence.eval(GenerateToken("+"))
        action_sequence.eval(GenerateToken("1"))
        action_sequence.eval(GenerateToken("2"))
        self.assertEqual(None, action_sequence.head)
        self.assertEqual(
            Node("def",
                 [Field("name", "value", Leaf("value", "f_0")),
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
            NodeType(Root(), NodeConstraint.Node),
            [("root", NodeType(Root(), NodeConstraint.Node))])))
        action_sequence.eval(ApplyRule(ExpandTreeRule(
            NodeType("op", NodeConstraint.Node), []
        )))
        self.assertEqual(Node("op", []), action_sequence.generate())

    def test_clone(self):
        action_sequence = ActionSequence()
        rule = ExpandTreeRule(NodeType("expr", NodeConstraint.Node),
                              [("elems",
                                NodeType("expr", NodeConstraint.Variadic))])
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
