from mlprogram.languages.csg.ast import (
    Circle,
    Difference,
    Rectangle,
    Reference,
    Rotation,
    Translation,
    Union,
)


class TestAST(object):
    def test_circle(self):
        assert "Circle" == Circle(1).type_name()

    def test_rectangle(self):
        assert "Rectangle" == Rectangle(1, 2).type_name()

    def test_translation(self):
        assert "Translation" == Translation(1, 2, Circle(3)).type_name()

    def test_rotation(self):
        assert "Rotation" == Rotation(45, Circle(3)).type_name()

    def test_union(self):
        assert "Union" == Union(Circle(2), Circle(3)).type_name()

    def test_difference(self):
        assert "Difference" == Difference(Circle(2), Circle(3)).type_name()

    def test_reference(self):
        assert "Reference" == Reference(0).type_name()
