
from mlprogram.languages import Field, Leaf, Node, Sugar


class TestLeaf(object):
    def test_clone(self):
        l0 = Leaf("type", "value")
        l1 = l0.clone()
        l1.type_name = "foo"
        l1.value = "bar"
        assert "type" == l0.type_name
        assert "value" == l0.value


class TestNode(object):
    def test_clone(self):
        a = Node("list",
                 [Field("name", "literal", Leaf("str", "name")),
                  Field("elems", "literal", [
                      Leaf("str", "foo"), Leaf("str", "bar")])])
        a1 = a.clone()
        a.name = ""
        a.type_name = ""
        a.fields[0].name = ""
        a.fields[1].value[0].type_name = ""
        assert Node("list",
                    [Field("name", "literal", Leaf("str", "name")),
                     Field("elems", "literal", [
                         Leaf("str", "foo"), Leaf("str", "bar")])
                     ]) == a1


class TestSugar(object):
    def test_node(self):
        assert Sugar.node("list", name=("literal", Leaf("str", "name"))) == \
            Node("list", [Field("name", "literal", Leaf("str", "name"))])

    def test_leaf(self):
        assert Sugar.leaf("str", "name") == Leaf("str", "name")
