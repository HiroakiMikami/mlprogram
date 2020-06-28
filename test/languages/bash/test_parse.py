import unittest

from mlprogram.asts import Node, Leaf, Field
from mlprogram.languages.bash import Parse, Unparse


class TestParse(unittest.TestCase):
    def test_simple_case(self):
        self.assertEqual(
            Node("Command",
                 [Field("parts", "Node", [
                     Node("Assign", [Field("value", "Node",
                                           [Node("Literal", [
                                               Field("value", "str",
                                                     Leaf("str", "x=10"))])])])
                 ])]),
            Parse()("x=10")
        )

    def test_invalid_case(self):
        self.assertEqual(None, Parse()("foobar`"))


class TestUnparse(unittest.TestCase):
    def test_simple_case(self):
        self.assertEqual("x=10", Unparse()(Parse()("x=10")))


if __name__ == "__main__":
    unittest.main()
