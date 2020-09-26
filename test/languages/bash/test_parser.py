import unittest

from mlprogram.languages import Node, Leaf, Field
from mlprogram.languages.bash import Parser


class TestParser(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(
            Node("Command",
                 [Field("parts", "Node", [
                     Node("Assign", [Field("value", "Node",
                                           [Node("Literal", [
                                               Field("value", "str",
                                                     [Leaf("str", "x=10")])])]
                                           )])
                 ])]),
            Parser(lambda x: [x]).parse("x=10")
        )

    def test_parse_invalid_code(self):
        self.assertEqual(None, Parser(lambda x: [x]).parse("foobar`"))

    def test_unparse(self):
        parser = Parser(lambda x: [x])
        self.assertEqual("x=10", parser.unparse(parser.parse("x=10")))


if __name__ == "__main__":
    unittest.main()
