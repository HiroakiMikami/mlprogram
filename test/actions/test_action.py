from mlprogram.actions \
    import ExpandTreeRule, GenerateToken, ApplyRule, NodeType, \
    NodeConstraint, CloseVariadicFieldRule


class TestNodeType(object):
    def test_str(self):
        assert "type" == str(NodeType("type", NodeConstraint.Node, False))
        assert "type*" == str(NodeType("type", NodeConstraint.Node, True))
        assert \
            "type(token)" == str(NodeType("type", NodeConstraint.Token, False))

    def test_eq(self):
        assert NodeType("foo", NodeConstraint.Node, False) == \
            NodeType("foo", NodeConstraint.Node, False)
        assert NodeType("foo", NodeConstraint.Node, False) != \
            NodeType("foo", NodeConstraint.Node, True)
        assert 0 != NodeType("foo", NodeConstraint.Node, False)


class TestRule(object):
    def test_str(self):
        t0 = NodeType("t0", NodeConstraint.Node, False)
        t1 = NodeType("t1", NodeConstraint.Node, False)
        t2 = NodeType("t2", NodeConstraint.Node, True)
        assert "t0 -> [elem0: t1, elem1: t2*]" == \
            str(ExpandTreeRule(t0, [("elem0", t1),
                                    ("elem1", t2)]))
        assert "<close variadic field>" == \
            str(CloseVariadicFieldRule())

    def test_eq(self):
        assert ExpandTreeRule(NodeType("foo", NodeConstraint.Node, False), [
            ("f0", NodeType("bar", NodeConstraint.Node, False))]) == \
            ExpandTreeRule(
                NodeType("foo", NodeConstraint.Node, False),
                [("f0", NodeType("bar", NodeConstraint.Node, False))])
        assert GenerateToken("", "foo") == GenerateToken("", "foo")
        assert ExpandTreeRule(NodeType("foo", NodeConstraint.Node, False), [
            ("f0", NodeType("bar", NodeConstraint.Node, False))]) != \
            ExpandTreeRule(NodeType("foo", NodeConstraint.Node, False), [])
        assert GenerateToken("", "foo") != GenerateToken("", "bar")
        assert ExpandTreeRule(
            NodeType("foo", NodeConstraint.Node, False),
            [
                ("f0", NodeType("bar", NodeConstraint.Node, False))
            ]) != GenerateToken("", "foo")
        assert 0 != ExpandTreeRule(
            NodeType("foo", NodeConstraint.Node, False),
            [("f0", NodeType("bar", NodeConstraint.Node, False))])


class TestAction(object):
    def test_str(self):
        t0 = NodeType("t0", NodeConstraint.Node, False)
        t1 = NodeType("t1", NodeConstraint.Node, False)
        t2 = NodeType("t2", NodeConstraint.Node, True)
        assert "Apply (t0 -> [elem0: t1, elem1: t2*])" == \
            str(ApplyRule(
                ExpandTreeRule(t0,
                               [("elem0", t1),
                                ("elem1", t2)])))

        assert "Generate bar:kind" == str(GenerateToken("kind", "bar"))
