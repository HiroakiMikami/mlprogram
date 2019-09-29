import unittest

from nl2code.language import action, ast


def tokenize(value: str):
    return value.split(" ")


class TestLeaf(unittest.TestCase):
    def test_to_action_sequence(self):
        self.assertEqual(
            [action.GenerateToken("t0"), action.GenerateToken(
                "t1"), action.GenerateToken(action.CloseNode())],
            ast.Leaf("str", "t0 t1").to_action_sequence(tokenize)
        )

    def test_clone(self):
        l0 = ast.Leaf("type", "value")
        l1 = l0.clone()
        l1.type_name = "foo"
        l1.value = "bar"
        self.assertEqual("type", l0.type_name)
        self.assertEqual("value", l0.value)


class TestNode(unittest.TestCase):
    def test_to_action_sequence(self):
        a = ast.Node(
            "def",
            [ast.Field("name", "literal", ast.Leaf("str", "foo"))])
        self.assertEqual(
            [action.ApplyRule(action.ExpandTreeRule(
                action.NodeType("def", action.NodeConstraint.Node),
                [("name",
                  action.NodeType("literal", action.NodeConstraint.Token))])),
             action.GenerateToken("foo"),
             action.GenerateToken(action.CloseNode())],
            a.to_action_sequence(tokenize)
        )

    def test_to_action_sequence_with_variadic_fields(self):
        a = ast.Node("list", [ast.Field("elems", "literal", [
                     ast.Node("str", []), ast.Node("str", [])])])
        self.assertEqual(
            [action.ApplyRule(action.ExpandTreeRule(
                action.NodeType("list", action.NodeConstraint.Node),
                [("elems",
                  action.NodeType("literal",
                                  action.NodeConstraint.Variadic))])),
             action.ApplyRule(action.ExpandTreeRule(
                 action.NodeType("str", action.NodeConstraint.Node),
                 [])),
             action.ApplyRule(action.ExpandTreeRule(
                 action.NodeType("str", action.NodeConstraint.Node),
                 [])),
             action.CloseVariadicFieldRule()],
            a.to_action_sequence(tokenize)
        )

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
