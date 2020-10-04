import unittest

from mlprogram.languages.csg.ast \
    import Circle, Rectangle, Translation, Rotation, Union, Difference


class TestAST(unittest.TestCase):
    def test_circle(self):
        self.assertEqual("Circle", Circle(1).type_name())

    def test_rectangle(self):
        self.assertEqual("Rectangle", Rectangle(1, 2).type_name())

    def test_translation(self):
        self.assertEqual("Translation",
                         Translation(1, 2, Circle(3)).type_name())

    def test_rotation(self):
        self.assertEqual("Rotation", Rotation(45, Circle(3)).type_name())

    def test_union(self):
        self.assertEqual("Union", Union(Circle(2), Circle(3)).type_name())

    def test_difference(self):
        self.assertEqual("Difference",
                         Difference(Circle(2), Circle(3)).type_name())


if __name__ == "__main__":
    unittest.main()
