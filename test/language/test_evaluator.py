import unittest

from nl2code.language.evaluator import Evaluator, Parent
from nl2code.language.evaluator import InvalidActionException
from nl2code.language.action import ExpandTreeRule, NodeType, NodeConstraint
from nl2code.language.action import ApplyRule, GenerateToken
from nl2code.language.action import CloseVariadicFieldRule, CloseNode
from nl2code.language.ast import Node, Leaf, Field


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
        self.assertEqual([ApplyRule(rule)], evaluator.action_sequence)
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
                         evaluator.action_sequence)
        self.assertEqual(Parent(0, 0), evaluator.parent(1))
        self.assertEqual([], evaluator._tree.children[1])

        evaluator.eval(GenerateToken("bar"))
        self.assertEqual(0, evaluator.head.action)
        self.assertEqual(0, evaluator.head.field)
        self.assertEqual([1, 2], evaluator._tree.children[0][0])
        self.assertEqual([ApplyRule(rule),
                          GenerateToken("foo"), GenerateToken("bar")],
                         evaluator.action_sequence)

        evaluator.eval(GenerateToken(CloseNode()))
        self.assertEqual(0, evaluator.head.action)
        self.assertEqual(1, evaluator.head.field)
        self.assertEqual([1, 2, 3], evaluator._tree.children[0][0])
        self.assertEqual([ApplyRule(rule),
                          GenerateToken("foo"), GenerateToken("bar"),
                          GenerateToken(CloseNode())],
                         evaluator.action_sequence)

        with self.assertRaises(InvalidActionException):
            evaluator.eval(GenerateToken("foo"))

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
                         evaluator.action_sequence)
        self.assertEqual(Parent(0, 0), evaluator.parent(1))
        self.assertEqual([], evaluator._tree.children[1])

        evaluator.eval(ApplyRule(rule0))
        self.assertEqual(0, evaluator.head.action)
        self.assertEqual(0, evaluator.head.field)
        self.assertEqual([1, 2], evaluator._tree.children[0][0])
        self.assertEqual([ApplyRule(rule), ApplyRule(rule0), ApplyRule(rule0)],
                         evaluator.action_sequence)

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

    def test_generate_ast(self):
        evaluator = Evaluator()
        funcdef = ExpandTreeRule(NodeType("def", NodeConstraint.Node),
                                 [("name",
                                   NodeType("value", NodeConstraint.Token)),
                                  ("body",
                                   NodeType("expr", NodeConstraint.Variadic))])
        expr = ExpandTreeRule(NodeType("expr", NodeConstraint.Node),
                              [("op", NodeType("value", NodeConstraint.Token)),
                               ("arg0",
                                NodeType("value", NodeConstraint.Token)),
                               ("arg1",
                                NodeType("value", NodeConstraint.Token))])
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
