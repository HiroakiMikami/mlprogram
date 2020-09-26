import unittest
from mlprogram.interpreters import Reference as R
from mlprogram.languages import Node, Leaf, Field
from mlprogram.languages.csg import Parser
from mlprogram.languages.csg import Circle, Rectangle, Reference
from mlprogram.languages.csg import Translation, Rotation
from mlprogram.languages.csg import Union, Difference


class TestParser(unittest.TestCase):
    def test_parse_circle(self):
        self.assertEqual(
            Node("Circle", [Field("r", "size", Leaf("size", 1))]),
            Parser().parse(Circle(1))
        )

    def test_parse_rectangle(self):
        self.assertEqual(
            Node("Rectangle", [
                Field("w", "size", Leaf("size", 1)),
                Field("h", "size", Leaf("size", 2))
            ]),
            Parser().parse(Rectangle(1, 2))
        )

    def test_parse_translation(self):
        self.assertEqual(
            Node("Translation", [
                Field("x", "length", Leaf("length", 1)),
                Field("y", "length", Leaf("length", 2)),
                Field("child", "CSG", Leaf("CSG", R(0)))
            ]),
            Parser().parse(Translation(1, 2, Reference(R(0))))
        )

    def test_parse_rotation(self):
        self.assertEqual(
            Node("Rotation", [
                Field("theta", "degree", Leaf("degree", 45)),
                Field("child", "CSG", Leaf("CSG", R(0)))
            ]),
            Parser().parse(Rotation(45, Reference(R(0))))
        )

    def test_parse_union(self):
        self.assertEqual(
            Node("Union", [
                Field("a", "CSG", Leaf("CSG", R(0))),
                Field("b", "CSG", Leaf("CSG", R(1)))
            ]),
            Parser().parse(Union(Reference(R(0)), Reference(R(1))))
        )

    def test_parse_difference(self):
        self.assertEqual(
            Node("Difference", [
                Field("a", "CSG", Leaf("CSG", R(0))),
                Field("b", "CSG", Leaf("CSG", R(1)))
            ]),
            Parser().parse(Difference(Reference(R(0)), Reference(R(1))))
        )

    def test_unparse_circle(self):
        self.assertEqual(
            Circle(1),
            Parser().unparse(Parser().parse(Circle(1)))
        )

    def test_unparse_rectangle(self):
        self.assertEqual(
            Rectangle(1, 2),
            Parser().unparse(Parser().parse(Rectangle(1, 2)))
        )

    def test_unparse_translation(self):
        self.assertEqual(
            Translation(1, 2, Reference(R(0))),
            Parser().unparse(
                Parser().parse(Translation(1, 2, Reference(R(0)))))
        )

    def test_unparse_rotation(self):
        self.assertEqual(
            Rotation(45, Reference(R(0))),
            Parser().unparse(Parser().parse(Rotation(45, Reference(R(0)))))
        )

    def test_unparse_union(self):
        self.assertEqual(
            Union(Reference(R(0)), Reference(R(1))),
            Parser().unparse(
                Parser().parse(Union(Reference(R(0)), Reference(R(1)))))
        )

    def test_unparse_difference(self):
        self.assertEqual(
            Difference(Reference(R(0)), Reference(R(1))),
            Parser().unparse(
                Parser().parse(Difference(Reference(R(0)), Reference(R(1))))
            )
        )


if __name__ == "__main__":
    unittest.main()
