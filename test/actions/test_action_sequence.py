import pytest
from mlprogram.actions.action_sequence import Parent
from mlprogram.actions import ActionSequence, InvalidActionException
from mlprogram.actions \
    import ExpandTreeRule, NodeType, NodeConstraint, ApplyRule, \
    GenerateToken, CloseVariadicFieldRule
from mlprogram.languages import Node, Leaf, Field, Root


class TestActionSequence(object):
    def test_eval_root(self):
        action_sequence = ActionSequence()
        assert action_sequence.head is None
        with pytest.raises(InvalidActionException):
            action_sequence = ActionSequence()
            action_sequence.eval(GenerateToken("kind", ""))
        with pytest.raises(InvalidActionException):
            action_sequence = ActionSequence()
            action_sequence.eval(ApplyRule(CloseVariadicFieldRule()))

        action_sequence = ActionSequence()
        rule = ExpandTreeRule(NodeType("def", NodeConstraint.Node, False),
                              [("name",
                                NodeType("value", NodeConstraint.Node, False)),
                               ("value",
                                NodeType("args", NodeConstraint.Node, True))])
        action_sequence.eval(ApplyRule(rule))
        assert 0 == action_sequence.head.action
        assert 0 == action_sequence.head.field
        assert [ApplyRule(rule)] == action_sequence.action_sequence
        assert action_sequence.parent(0) is None
        assert [[], []] == action_sequence._tree.children[0]

    def test_generate_variadic_token(self):
        action_sequence = ActionSequence()
        rule = ExpandTreeRule(
            NodeType("def", NodeConstraint.Node, False),
            [("name",
              NodeType("value", NodeConstraint.Token, True)),
             ("value",
              NodeType("args", NodeConstraint.Node, True))])
        action_sequence.eval(ApplyRule(rule))
        action_sequence.eval(GenerateToken("", "foo"))
        assert 0 == action_sequence.head.action
        assert 0 == action_sequence.head.field
        assert [1] == action_sequence._tree.children[0][0]
        assert [ApplyRule(rule),
                GenerateToken("", "foo")] == action_sequence.action_sequence
        assert Parent(0, 0) == action_sequence.parent(1)
        assert [] == action_sequence._tree.children[1]

        action_sequence.eval(GenerateToken("", "bar"))
        assert 0 == action_sequence.head.action
        assert 0 == action_sequence.head.field
        assert [1, 2] == action_sequence._tree.children[0][0]
        assert [ApplyRule(rule),
                GenerateToken("", "foo"),
                GenerateToken("", "bar")] == action_sequence.action_sequence

        action_sequence.eval(ApplyRule(CloseVariadicFieldRule()))
        assert 0 == action_sequence.head.action
        assert 1 == action_sequence.head.field
        assert [1, 2, 3] == action_sequence._tree.children[0][0]
        assert [ApplyRule(rule),
                GenerateToken("", "foo"),
                GenerateToken("", "bar"),
                ApplyRule(CloseVariadicFieldRule())
                ] == action_sequence.action_sequence

        with pytest.raises(InvalidActionException):
            action_sequence.eval(GenerateToken("", "foo"))

    def test_generate_token(self):
        action_sequence = ActionSequence()
        rule = ExpandTreeRule(
            NodeType("def", NodeConstraint.Node, False),
            [("name",
              NodeType("value", NodeConstraint.Token, False)),
             ("value",
              NodeType("args", NodeConstraint.Node, True))])
        action_sequence.eval(ApplyRule(rule))
        action_sequence.eval(GenerateToken("", "foo"))
        assert 0 == action_sequence.head.action
        assert 1 == action_sequence.head.field
        assert [1] == action_sequence._tree.children[0][0]
        assert [ApplyRule(rule),
                GenerateToken("", "foo")] == action_sequence.action_sequence
        assert Parent(0, 0) == action_sequence.parent(1)
        assert [] == action_sequence._tree.children[1]

        with pytest.raises(InvalidActionException):
            action_sequence.eval(GenerateToken("", "bar"))

        action_sequence = ActionSequence()
        action_sequence.eval(ApplyRule(rule))
        with pytest.raises(InvalidActionException):
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
        assert 0 == action_sequence.head.action
        assert 0 == action_sequence.head.field
        assert [1] == action_sequence._tree.children[0][0]
        assert [ApplyRule(rule),
                ApplyRule(rule0)] == action_sequence.action_sequence
        assert Parent(0, 0) == action_sequence.parent(1)
        assert [] == action_sequence._tree.children[1]

        action_sequence.eval(ApplyRule(rule0))
        assert 0 == action_sequence.head.action
        assert 0 == action_sequence.head.field
        assert [1, 2] == action_sequence._tree.children[0][0]
        assert [ApplyRule(rule),
                ApplyRule(rule0),
                ApplyRule(rule0)] == action_sequence.action_sequence

        action_sequence.eval(ApplyRule(CloseVariadicFieldRule()))
        assert action_sequence.head is None

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
        assert 0 == action_sequence.head.action
        assert 1 == action_sequence.head.field

    def test_invalid_close_variadic_field_rule(self):
        rule = ExpandTreeRule(NodeType("expr", NodeConstraint.Node, False),
                              [("elems",
                                NodeType("value", NodeConstraint.Node, False))]
                              )

        action_sequence = ActionSequence()
        action_sequence.eval(ApplyRule(rule))
        with pytest.raises(InvalidActionException):
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
        action_sequence.eval(GenerateToken("name", "f"))
        action_sequence.eval(GenerateToken("name", "_"))
        action_sequence.eval(GenerateToken("name", "0"))
        action_sequence.eval(ApplyRule(CloseVariadicFieldRule()))
        action_sequence.eval(ApplyRule(expr))
        action_sequence.eval(GenerateToken("value", "+"))
        action_sequence.eval(GenerateToken("value", "1"))
        action_sequence.eval(GenerateToken("value", "2"))
        action_sequence.eval(ApplyRule(CloseVariadicFieldRule()))
        assert action_sequence.head is None
        assert Node("def",
                    [Field("name", "value", [Leaf("name", "f"),
                                             Leaf("name", "_"),
                                             Leaf("name", "0")]),
                     Field("body", "expr", [
                         Node("expr",
                              [
                                  Field("op", "value", Leaf("value", "+")),
                                  Field("arg0", "value", Leaf("value", "1")),
                                  Field("arg1", "value", Leaf("value", "2"))
                              ])
                     ])]) == action_sequence.generate()

    def test_generate_ignore_root_type(self):
        action_sequence = ActionSequence()
        action_sequence.eval(ApplyRule(ExpandTreeRule(
            NodeType(Root(), NodeConstraint.Node, False),
            [("root", NodeType(Root(), NodeConstraint.Node, False))])))
        action_sequence.eval(ApplyRule(ExpandTreeRule(
            NodeType("op", NodeConstraint.Node, False), []
        )))
        assert Node("op", []) == action_sequence.generate()

    def test_clone(self):
        action_sequence = ActionSequence()
        rule = ExpandTreeRule(NodeType("expr", NodeConstraint.Node, False),
                              [("elems",
                                NodeType("expr", NodeConstraint.Node, True))])
        action_sequence.eval(ApplyRule(rule))

        action_sequence2 = action_sequence.clone()
        assert action_sequence.generate() == action_sequence2.generate()

        action_sequence2.eval(ApplyRule(rule))
        assert \
            action_sequence._tree.children != action_sequence2._tree.children
        assert \
            action_sequence._tree.parent != action_sequence2._tree.parent
        assert \
            action_sequence.action_sequence != action_sequence2.action_sequence
        assert action_sequence._head_action_index != \
            action_sequence2._head_action_index
        assert action_sequence._head_children_index != \
            action_sequence2._head_children_index
        assert action_sequence.generate() != action_sequence2.generate()

    def test_create_leaf(self):
        seq = ActionSequence.create(Leaf("str", "t0 t1"))
        assert [ApplyRule(ExpandTreeRule(
                NodeType(None, NodeConstraint.Node, False),
                [("root", NodeType(Root(),
                                   NodeConstraint.Token, False))
                 ])),
                GenerateToken("str", "t0 t1")] == seq.action_sequence

        seq = ActionSequence.create(Node("value", [
            Field("name", "str",
                  [Leaf("str", "t0"), Leaf("str", "t1")])]))
        assert [ApplyRule(ExpandTreeRule(
                NodeType(None, NodeConstraint.Node, False),
                [("root", NodeType(Root(),
                                   NodeConstraint.Node, False))
                 ])),
                ApplyRule(ExpandTreeRule(
                    NodeType("value", NodeConstraint.Node, False),
                    [("name", NodeType("str", NodeConstraint.Token, True))]
                )),
                GenerateToken("str", "t0"),
                GenerateToken("str", "t1"),
                ApplyRule(CloseVariadicFieldRule())] == seq.action_sequence

    def test_create_node(self):
        a = Node(
            "def",
            [Field("name", "literal", Leaf("str", "foo"))])
        seq = ActionSequence.create(a)
        assert [ApplyRule(ExpandTreeRule(
                NodeType(None, NodeConstraint.Node, False),
                [("root", NodeType(Root(),
                                   NodeConstraint.Node, False))
                 ])),
                ApplyRule(ExpandTreeRule(
                    NodeType("def", NodeConstraint.Node, False),
                    [("name",
                      NodeType("literal", NodeConstraint.Token, False))])),
                GenerateToken("str", "foo")] == seq.action_sequence

    def test_create_node_with_variadic_fields(self):
        a = Node("list", [Field("elems", "literal", [
            Node("str", []), Node("str", [])])])
        seq = ActionSequence.create(a)
        assert [ApplyRule(ExpandTreeRule(
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
                ApplyRule(CloseVariadicFieldRule())] == seq.action_sequence
