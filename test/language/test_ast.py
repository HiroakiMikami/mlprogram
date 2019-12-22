import unittest

from nl2prog.language import ast


class TestLeaf(unittest.TestCase):
    def test_clone(self):
        l0 = ast.Leaf("type", "value")
        l1 = l0.clone()
        l1.type_name = "foo"
        l1.value = "bar"
        self.assertEqual("type", l0.type_name)
        self.assertEqual("value", l0.value)


class TestNode(unittest.TestCase):
    def test_clone(self):
        a = ast.Node("list",
                     [ast.Field("name", "literal", ast.Leaf("str", "name")),
                      ast.Field("elems", "literal", [
                          ast.Leaf("str", "foo"), ast.Leaf("str", "bar")])])
        a1 = a.clone()
        a.name = ""
        a.type_name = ""
        a.fields[0].name = ""
        a.fields[1].value[0].type_name = ""
        self.assertEqual(
            ast.Node("list",
                     [ast.Field("name", "literal", ast.Leaf("str", "name")),
                      ast.Field("elems", "literal", [
                          ast.Leaf("str", "foo"), ast.Leaf("str", "bar")])]),
            a1)


if __name__ == "__main__":
    unittest.main()
