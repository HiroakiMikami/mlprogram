import unittest

from mlprogram.language.csg.ast \
    import Circle, Rectangle, Translation, Rotation, Union, Difference
from mlprogram.ast.ast import Node, Field, Leaf


class TestAst(unittest.TestCase):
    def test_circle(self):
        self.assertEqual(
            Node("Circle", [Field("r", "number", Leaf("number", "1"))]),
            Circle(1)
        )

    def test_rectangle(self):
        self.assertEqual(
            Node("Rectangle", [
                Field("width", "number", Leaf("number", "1")),
                Field("height", "number", Leaf("number", "2"))
            ]),
            Rectangle(1, 2)
        )

    def test_translation(self):
        self.assertEqual(
            Node("Translation", [
                Field("x", "number", Leaf("number", "1")),
                Field("y", "number", Leaf("number", "2")),
                Field("child", "CSG", Circle(3))
            ]),
            Translation(1, 2, Circle(3))
        )

    def test_rotation(self):
        self.assertEqual(
            Node("Rotation", [
                Field("theta", "number", Leaf("number", "45")),
                Field("child", "CSG", Circle(3))
            ]),
            Rotation(45, Circle(3))
        )

    def test_union(self):
        self.assertEqual(
            Node("Union", [
                Field("a", "CSG", Circle(2)),
                Field("b", "CSG", Circle(3))
            ]),
            Union(Circle(2), Circle(3))
        )

    def test_difference(self):
        self.assertEqual(
            Node("Difference", [
                Field("a", "CSG", Circle(2)),
                Field("b", "CSG", Circle(3))
            ]),
            Difference(Circle(2), Circle(3))
        )


if __name__ == "__main__":
    unittest.main()
