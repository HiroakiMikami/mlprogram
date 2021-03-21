import pytest

from mlprogram.languages import Field, Leaf, Node
from mlprogram.languages.csg import (
    Circle,
    Difference,
    Parser,
    Rectangle,
    Reference,
    Rotation,
    Translation,
    Union,
)


@pytest.fixture
def parser():
    return Parser()


class TestParser(object):
    def test_parse_circle(self, parser):
        assert Node("Circle",
                    [Field("r", "size", Leaf("size", 1))]
                    ) == parser.parse(Circle(1))

    def test_parse_rectangle(self, parser):
        assert Node("Rectangle", [
            Field("w", "size", Leaf("size", 1)),
            Field("h", "size", Leaf("size", 2))
        ]) == parser.parse(Rectangle(1, 2))

    def test_parse_translation(self, parser):
        assert Node("Translation", [
            Field("x", "length", Leaf("length", 1)),
            Field("y", "length", Leaf("length", 2)),
            Field("child", "CSG", Leaf("CSG", Reference(0)))
        ]) == parser.parse(Translation(1, 2, Reference(0)))

    def test_parse_rotation(self, parser):
        assert Node("Rotation", [
            Field("theta", "degree", Leaf("degree", 45)),
            Field("child", "CSG", Leaf("CSG", Reference(0)))
        ]) == parser.parse(Rotation(45, Reference(0)))

    def test_parse_union(self, parser):
        assert Node("Union", [
            Field("a", "CSG", Leaf("CSG", Reference(0))),
            Field("b", "CSG", Leaf("CSG", Reference(1)))
        ]) == parser.parse(Union(Reference(0), Reference(1)))

    def test_parse_difference(self, parser):
        assert Node("Difference", [
            Field("a", "CSG", Leaf("CSG", Reference(0))),
            Field("b", "CSG", Leaf("CSG", Reference(1)))
        ]) == parser.parse(Difference(Reference(0), Reference(1)))

    def test_unparse_circle(self, parser):
        assert Circle(1) == parser.unparse(parser.parse(Circle(1)))

    def test_unparse_rectangle(self, parser):
        assert Rectangle(1, 2) == \
            parser.unparse(parser.parse(Rectangle(1, 2)))

    def test_unparse_translation(self, parser):
        assert Translation(1, 2, Reference(0)) == parser.unparse(
            parser.parse(Translation(1, 2, Reference(0)))
        )

    def test_unparse_rotation(self, parser):
        assert Rotation(45, Reference(0)) == parser.unparse(
            parser.parse(Rotation(45, Reference(0)))
        )

    def test_unparse_union(self, parser):
        assert Union(Reference(0), Reference(1)) == \
            parser.unparse(
                parser.parse(Union(Reference(0), Reference(1)))
        )

    def test_unparse_difference(self, parser):
        assert Difference(Reference(0),
                          Reference(1)) == \
            parser.unparse(
                parser.parse(Difference(Reference(0), Reference(1)))
        )
