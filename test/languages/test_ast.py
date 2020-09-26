import unittest

from mlprogram.languages import Node
from mlprogram.languages import Field
from mlprogram.languages import Leaf


class TestLeaf(unittest.TestCase):
    def test_clone(self):
        l0 = Leaf("type", "value")
        l1 = l0.clone()
        l1.type_name = "foo"
        l1.value = "bar"
        self.assertEqual("type", l0.type_name)
        self.assertEqual("value", l0.value)


class TestNode(unittest.TestCase):
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
        self.assertEqual(
            Node("list",
                 [Field("name", "literal", Leaf("str", "name")),
                  Field("elems", "literal", [
                      Leaf("str", "foo"), Leaf("str", "bar")])
                  ]),
            a1)


if __name__ == "__main__":
    unittest.main()
