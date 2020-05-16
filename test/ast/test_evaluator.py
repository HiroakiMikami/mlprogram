import unittest

from mlprogram.action.evaluator \
    import Evaluator, Parent, InvalidActionException
from mlprogram.action.action \
    import ExpandTreeRule, NodeType, NodeConstraint, ApplyRule, \
    GenerateToken, CloseVariadicFieldRule, CloseNode, ActionOptions
from mlprogram.action.ast import Node, Leaf, Field, Root


class TestEvaluator(unittest.TestCase):
    def test_eval_root(self):
        evaluator = Evaluator()
        self.assertEqual(None, evaluator.head)
        with self.assertRaises(InvalidActionException):
            evaluator = Evaluator()
            evaluator.eval(GenerateToken(""))
        with self.assertRaises(InvalidActionException):
            evaluator = Evaluator()
            evaluator.eval(ApplyRule(CloseVariadicFieldRule()))

        evaluator = Evaluator()
        rule = ExpandTreeRule(NodeType("def", NodeConstraint.Node),
                              [("name",
                                NodeType("value", NodeConstraint.Node)),
                               ("value",
                                NodeType("args", NodeConstraint.Variadic))])
        evaluator.eval(ApplyRule(rule))
        self.assertEqual(0, evaluator.head.action)
        self.assertEqual(0, evaluator.head.field)
        self.assertEqual([ApplyRule(rule)], evaluator.action_sequence.sequence)
        self.assertEqual(None, evaluator.parent(0))
        self.assertEqual([[], []], evaluator._tree.children[0])

    def test_generate_token(self):
        evaluator = Evaluator()
        rule = ExpandTreeRule(NodeType("def", NodeConstraint.Node),
                              [("name",
                                NodeType("value", NodeConstraint.Token)),
                               ("value",
                                NodeType("args", NodeConstraint.Variadic))])
        evaluator.eval(ApplyRule(rule))
        evaluator.eval(GenerateToken("foo"))
        self.assertEqual(0, evaluator.head.action)
        self.assertEqual(0, evaluator.head.field)
        self.assertEqual([1], evaluator._tree.children[0][0])
        self.assertEqual([ApplyRule(rule), GenerateToken("foo")],
                         evaluator.action_sequence.sequence)
        self.assertEqual(Parent(0, 0), evaluator.parent(1))
        self.assertEqual([], evaluator._tree.children[1])

        evaluator.eval(GenerateToken("bar"))
        self.assertEqual(0, evaluator.head.action)
        self.assertEqual(0, evaluator.head.field)
        self.assertEqual([1, 2], evaluator._tree.children[0][0])
        self.assertEqual([ApplyRule(rule),
                          GenerateToken("foo"), GenerateToken("bar")],
                         evaluator.action_sequence.sequence)

        evaluator.eval(GenerateToken(CloseNode()))
        self.assertEqual(0, evaluator.head.action)
        self.assertEqual(1, evaluator.head.field)
        self.assertEqual([1, 2, 3], evaluator._tree.children[0][0])
        self.assertEqual([ApplyRule(rule),
                          GenerateToken("foo"), GenerateToken("bar"),
                          GenerateToken(CloseNode())],
                         evaluator.action_sequence.sequence)

        with self.assertRaises(InvalidActionException):
            evaluator.eval(GenerateToken("foo"))

    def test_generate_token_with_split_non_terminal_False(self):
        evaluator = Evaluator(ActionOptions(True, False))
        rule = ExpandTreeRule(NodeType("def", NodeConstraint.Node),
                              [("name",
                                NodeType("value", NodeConstraint.Token)),
                               ("value",
                                NodeType("args", NodeConstraint.Variadic))])
        evaluator.eval(ApplyRule(rule))
        evaluator.eval(GenerateToken("foo"))
        self.assertEqual(0, evaluator.head.action)
        self.assertEqual(1, evaluator.head.field)
        self.assertEqual([1], evaluator._tree.children[0][0])
        self.assertEqual([ApplyRule(rule), GenerateToken("foo")],
                         evaluator.action_sequence.sequence)
        self.assertEqual(Parent(0, 0), evaluator.parent(1))
        self.assertEqual([], evaluator._tree.children[1])

        with self.assertRaises(InvalidActionException):
            evaluator.eval(GenerateToken("bar"))

        evaluator = Evaluator(ActionOptions(True, False))
        evaluator.eval(ApplyRule(rule))
        with self.assertRaises(InvalidActionException):
            evaluator.eval(GenerateToken(CloseNode()))

    def test_variadic_field(self):
        evaluator = Evaluator()
        rule = ExpandTreeRule(NodeType("expr", NodeConstraint.Node),
                              [("elems",
                                NodeType("value", NodeConstraint.Variadic))])
        rule0 = ExpandTreeRule(NodeType("value", NodeConstraint.Node),
                               [])
        evaluator.eval(ApplyRule(rule))
        evaluator.eval(ApplyRule(rule0))
        self.assertEqual(0, evaluator.head.action)
        self.assertEqual(0, evaluator.head.field)
        self.assertEqual([1], evaluator._tree.children[0][0])
        self.assertEqual([ApplyRule(rule), ApplyRule(rule0)],
                         evaluator.action_sequence.sequence)
        self.assertEqual(Parent(0, 0), evaluator.parent(1))
        self.assertEqual([], evaluator._tree.children[1])

        evaluator.eval(ApplyRule(rule0))
        self.assertEqual(0, evaluator.head.action)
        self.assertEqual(0, evaluator.head.field)
        self.assertEqual([1, 2], evaluator._tree.children[0][0])
        self.assertEqual([ApplyRule(rule), ApplyRule(rule0), ApplyRule(rule0)],
                         evaluator.action_sequence.sequence)

        evaluator.eval(ApplyRule(CloseVariadicFieldRule()))
        self.assertEqual(None, evaluator.head)

        evaluator = Evaluator()
        rule1 = ExpandTreeRule(NodeType("expr", NodeConstraint.Node),
                               [("elems",
                                 NodeType("value", NodeConstraint.Variadic)),
                                ("name",
                                 NodeType("value", NodeConstraint.Node))])
        rule0 = ExpandTreeRule(NodeType("value", NodeConstraint.Node),
                               [])
        evaluator.eval(ApplyRule(rule1))
        evaluator.eval(ApplyRule(rule0))
        evaluator.eval(ApplyRule(CloseVariadicFieldRule))
        self.assertEqual(0, evaluator.head.action)
        self.assertEqual(1, evaluator.head.field)

    def test_variadic_field_retain_variadic_fields_False(self):
        evaluator = Evaluator(ActionOptions(False, True))
        rule = ExpandTreeRule(NodeType("expr", NodeConstraint.Node),
                              [("elems",
                                NodeType("value", NodeConstraint.Variadic))])
        rule0 = ExpandTreeRule(NodeType("value", NodeConstraint.Variadic),
                               [("0", NodeType("value", NodeConstraint.Node)),
                                ("1", NodeType("value", NodeConstraint.Node))])
        rule1 = ExpandTreeRule(NodeType("value", NodeConstraint.Node),
                               [])
        evaluator.eval(ApplyRule(rule))
        self.assertEqual(0, evaluator.head.action)
        self.assertEqual(0, evaluator.head.field)
        self.assertEqual([], evaluator._tree.children[0][0])
        self.assertEqual([ApplyRule(rule)],
                         evaluator.action_sequence.sequence)
        evaluator.eval(ApplyRule(rule0))
        self.assertEqual(1, evaluator.head.action)
        self.assertEqual(0, evaluator.head.field)
        self.assertEqual([1], evaluator._tree.children[0][0])
        self.assertEqual(Parent(0, 0), evaluator.parent(1))
        self.assertEqual([ApplyRule(rule), ApplyRule(rule0)],
                         evaluator.action_sequence.sequence)
        evaluator.eval(ApplyRule(rule1))
        self.assertEqual(1, evaluator.head.action)
        self.assertEqual(1, evaluator.head.field)
        self.assertEqual([2], evaluator._tree.children[1][0])
        self.assertEqual([], evaluator._tree.children[2])
        self.assertEqual(Parent(1, 0), evaluator.parent(2))
        self.assertEqual([ApplyRule(rule), ApplyRule(rule0), ApplyRule(rule1)],
                         evaluator.action_sequence.sequence)
        evaluator.eval(ApplyRule(rule1))
        self.assertEqual(None, evaluator.head)
        self.assertEqual([3], evaluator._tree.children[1][1])
        self.assertEqual([], evaluator._tree.children[3])
        self.assertEqual(Parent(1, 1), evaluator.parent(3))
        self.assertEqual([ApplyRule(rule), ApplyRule(rule0), ApplyRule(rule1),
                          ApplyRule(rule1)],
                         evaluator.action_sequence.sequence)

        evaluator = Evaluator(ActionOptions(False, True))
        rule = ExpandTreeRule(NodeType("expr", NodeConstraint.Node),
                              [("elems",
                                NodeType("value", NodeConstraint.Variadic)),
                               ("name",
                                NodeType("value", NodeConstraint.Node))])
        evaluator.eval(ApplyRule(rule))
        evaluator.eval(ApplyRule(rule0))
        evaluator.eval(ApplyRule(rule1))
        evaluator.eval(ApplyRule(rule1))
        self.assertEqual(0, evaluator.head.action)
        self.assertEqual(1, evaluator.head.field)

        evaluator = Evaluator(ActionOptions(False, True))
        with self.assertRaises(InvalidActionException):
            evaluator.eval(ApplyRule(CloseVariadicFieldRule()))

    def test_generate_ast(self):
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

        evaluator = Evaluator()
        evaluator.eval(ApplyRule(funcdef))
        evaluator.eval(GenerateToken("f"))
        evaluator.eval(GenerateToken("_"))
        evaluator.eval(GenerateToken("0"))
        evaluator.eval(GenerateToken(CloseNode()))
        evaluator.eval(ApplyRule(expr))
        evaluator.eval(GenerateToken("+"))
        evaluator.eval(GenerateToken(CloseNode()))
        evaluator.eval(GenerateToken("1"))
        evaluator.eval(GenerateToken(CloseNode()))
        evaluator.eval(GenerateToken("2"))
        evaluator.eval(GenerateToken(CloseNode()))
        evaluator.eval(ApplyRule(CloseVariadicFieldRule()))
        self.assertEqual(None, evaluator.head)
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
            evaluator.generate_ast()
        )

        evaluator = Evaluator(ActionOptions(False, True))
        evaluator.eval(ApplyRule(funcdef))
        evaluator.eval(GenerateToken("f_0"))
        evaluator.eval(GenerateToken(CloseNode()))
        evaluator.eval(ApplyRule(expr_expand))
        evaluator.eval(ApplyRule(expr))
        evaluator.eval(GenerateToken("+"))
        evaluator.eval(GenerateToken(CloseNode()))
        evaluator.eval(GenerateToken("1"))
        evaluator.eval(GenerateToken(CloseNode()))
        evaluator.eval(GenerateToken("2"))
        evaluator.eval(GenerateToken(CloseNode()))
        self.assertEqual(None, evaluator.head)
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
            evaluator.generate_ast()
        )

        evaluator = Evaluator(ActionOptions(True, False))
        evaluator.eval(ApplyRule(funcdef))
        evaluator.eval(GenerateToken("f_0"))
        evaluator.eval(ApplyRule(expr))
        evaluator.eval(GenerateToken("+"))
        evaluator.eval(GenerateToken("1"))
        evaluator.eval(GenerateToken("2"))
        evaluator.eval(ApplyRule(CloseVariadicFieldRule()))
        self.assertEqual(None, evaluator.head)
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
            evaluator.generate_ast()
        )

        evaluator = Evaluator(ActionOptions(False, False))
        evaluator.eval(ApplyRule(funcdef))
        evaluator.eval(GenerateToken("f_0"))
        evaluator.eval(ApplyRule(expr_expand))
        evaluator.eval(ApplyRule(expr))
        evaluator.eval(GenerateToken("+"))
        evaluator.eval(GenerateToken("1"))
        evaluator.eval(GenerateToken("2"))
        self.assertEqual(None, evaluator.head)
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
            evaluator.generate_ast()
        )

    def test_generate_ast_ignore_root_type(self):
        evaluator = Evaluator()
        evaluator.eval(ApplyRule(ExpandTreeRule(
            NodeType(Root(), NodeConstraint.Node),
            [("root", NodeType(Root(), NodeConstraint.Node))])))
        evaluator.eval(ApplyRule(ExpandTreeRule(
            NodeType("op", NodeConstraint.Node), []
        )))
        self.assertEqual(Node("op", []), evaluator.generate_ast())

    def test_clone(self):
        evaluator = Evaluator()
        rule = ExpandTreeRule(NodeType("expr", NodeConstraint.Node),
                              [("elems",
                                NodeType("expr", NodeConstraint.Variadic))])
        evaluator.eval(ApplyRule(rule))

        evaluator2 = evaluator.clone()
        self.assertEqual(evaluator.generate_ast(), evaluator2.generate_ast())

        evaluator2.eval(ApplyRule(rule))
        self.assertNotEqual(evaluator._tree.children,
                            evaluator2._tree.children)
        self.assertNotEqual(evaluator._tree.parent,
                            evaluator2._tree.parent)
        self.assertNotEqual(evaluator.action_sequence,
                            evaluator2.action_sequence)
        self.assertNotEqual(evaluator._head_action_index,
                            evaluator2._head_action_index)
        self.assertNotEqual(evaluator._head_children_index,
                            evaluator2._head_children_index)
        self.assertNotEqual(evaluator.generate_ast(),
                            evaluator2.generate_ast())


if __name__ == "__main__":
    unittest.main()
