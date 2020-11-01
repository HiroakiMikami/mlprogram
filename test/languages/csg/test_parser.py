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


class TestParser(object):
    def test_parse_circle(self):
        assert Node("Circle",
                    [Field("r", "size", Leaf("size", 1))]
                    ) == Parser().parse(Circle(1))

    def test_parse_rectangle(self):
        assert Node("Rectangle", [
            Field("w", "size", Leaf("size", 1)),
            Field("h", "size", Leaf("size", 2))
        ]) == Parser().parse(Rectangle(1, 2))

    def test_parse_translation(self):
        assert Node("Translation", [
            Field("x", "length", Leaf("length", 1)),
            Field("y", "length", Leaf("length", 2)),
            Field("child", "CSG", Leaf("CSG", Circle(1)))
        ]) == Parser().parse(Translation(1, 2, Reference(Circle(1))))

    def test_parse_rotation(self):
        assert Node("Rotation", [
            Field("theta", "degree", Leaf("degree", 45)),
            Field("child", "CSG", Leaf("CSG", Rectangle(1, 1)))
        ]) == Parser().parse(Rotation(45, Reference(Rectangle(1, 1))))

    def test_parse_union(self):
        assert Node("Union", [
            Field("a", "CSG", Leaf("CSG", Circle(1))),
            Field("b", "CSG", Leaf("CSG", Circle(1)))
        ]) == Parser().parse(Union(Reference(Circle(1)), Reference(Circle(1))))

    def test_parse_difference(self):
        assert Node("Difference", [
            Field("a", "CSG", Leaf("CSG", Circle(1))),
            Field("b", "CSG", Leaf("CSG", Circle(1)))
        ]) == Parser().parse(Difference(Reference(Circle(1)),
                                        Reference(Circle(1))))

    def test_unparse_circle(self):
        assert Circle(1) == Parser().unparse(Parser().parse(Circle(1)))

    def test_unparse_rectangle(self):
        assert Rectangle(1, 2) == \
            Parser().unparse(Parser().parse(Rectangle(1, 2)))

    def test_unparse_translation(self):
        assert Translation(1, 2, Reference(Circle(1))) == Parser().unparse(
            Parser().parse(Translation(1, 2, Reference(Circle(1)))))

    def test_unparse_rotation(self):
        assert Rotation(45, Reference(Circle(1))) == Parser().unparse(
            Parser().parse(Rotation(45, Reference(Circle(1)))))

    def test_unparse_union(self):
        assert Union(Reference(Circle(1)),
                     Reference(Rectangle(1, 1))) == \
            Parser().unparse(
                Parser().parse(Union(Reference(Circle(1)),
                                     Reference(Rectangle(1, 1)))))

    def test_unparse_difference(self):
        assert Difference(Reference(Circle(1)),
                          Reference(Rectangle(1, 1))) == \
            Parser().unparse(
                Parser().parse(Difference(Reference(Circle(1)),
                                          Reference(Rectangle(1, 1))))
        )
