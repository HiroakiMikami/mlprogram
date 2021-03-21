
from mlprogram.languages import Field, Leaf, Node
from mlprogram.languages.bash import Parser


class TestParser(object):
    def test_parse(self):
        assert Node("Command",
                    [Field("parts", "Node", [
                     Node("Assign", [Field("value", "Node",
                                           [Node("Literal", [
                                               Field("value", "str",
                                                     [Leaf("str", "x=10")])])]
                                           )])
                     ])]) == Parser(lambda x: [x]).parse("x=10")

    def test_parse_invalid_code(self):
        assert Parser(lambda x: [x]).parse("foobar`") is None

    def test_unparse(self):
        parser = Parser(lambda x: [x])
        assert "x=10" == parser.unparse(parser.parse("x=10"))
