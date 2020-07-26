import unittest
from mlprogram.utils import Reference as R
from mlprogram.asts import Node, Leaf, Field
from mlprogram.languages.csg import GetTokenType, IsSubtype, ToAst, ToCsgAst
from mlprogram.languages.csg import Circle, Rectangle, Reference
from mlprogram.languages.csg import Translation, Rotation
from mlprogram.languages.csg import Union, Difference
from mlprogram.languages.csg import get_samples, Dataset
from mlprogram.utils.transform import AstToSingleActionSequence


class TestGetTokenType(unittest.TestCase):
    def test_csg_type(self):
        self.assertEqual("CSG", GetTokenType()(R("0")))

    def test_number(self):
        self.assertEqual("int", GetTokenType()(0))


class TestIsSubType(unittest.TestCase):
    def test_happy_path(self):
        self.assertTrue(IsSubtype()("CSG", "CSG"))

    def test_integer(self):
        self.assertTrue(IsSubtype()("int", "size"))
        self.assertFalse(IsSubtype()("int", "CSG"))


class TestToAst(unittest.TestCase):
    def test_circle(self):
        self.assertEqual(
            Node("Circle", [Field("r", "size", Leaf("size", 1))]),
            ToAst()(Circle(1))
        )

    def test_rectangle(self):
        self.assertEqual(
            Node("Rectangle", [
                Field("w", "size", Leaf("size", 1)),
                Field("h", "size", Leaf("size", 2))
            ]),
            ToAst()(Rectangle(1, 2))
        )

    def test_translation(self):
        self.assertEqual(
            Node("Translation", [
                Field("x", "length", Leaf("length", 1)),
                Field("y", "length", Leaf("length", 2)),
                Field("child", "CSG", Leaf("CSG", R(0)))
            ]),
            ToAst()(Translation(1, 2, Reference(R(0))))
        )

    def test_rotation(self):
        self.assertEqual(
            Node("Rotation", [
                Field("theta", "degree", Leaf("degree", 45)),
                Field("child", "CSG", Leaf("CSG", R(0)))
            ]),
            ToAst()(Rotation(45, Reference(R(0))))
        )

    def test_union(self):
        self.assertEqual(
            Node("Union", [
                Field("a", "CSG", Leaf("CSG", R(0))),
                Field("b", "CSG", Leaf("CSG", R(1)))
            ]),
            ToAst()(Union(Reference(R(0)), Reference(R(1))))
        )

    def test_difference(self):
        self.assertEqual(
            Node("Difference", [
                Field("a", "CSG", Leaf("CSG", R(0))),
                Field("b", "CSG", Leaf("CSG", R(1)))
            ]),
            ToAst()(Difference(Reference(R(0)), Reference(R(1))))
        )


class TestToCsgAst(unittest.TestCase):
    def test_circle(self):
        self.assertEqual(
            Circle(1),
            ToCsgAst()(ToAst()(Circle(1)))
        )

    def test_rectangle(self):
        self.assertEqual(
            Rectangle(1, 2),
            ToCsgAst()(ToAst()(Rectangle(1, 2)))
        )

    def test_translation(self):
        self.assertEqual(
            Translation(1, 2, Reference(R(0))),
            ToCsgAst()(ToAst()(Translation(1, 2, Reference(R(0)))))
        )

    def test_rotation(self):
        self.assertEqual(
            Rotation(45, Reference(R(0))),
            ToCsgAst()(ToAst()(Rotation(45, Reference(R(0)))))
        )

    def test_union(self):
        self.assertEqual(
            Union(Reference(R(0)), Reference(R(1))),
            ToCsgAst()(ToAst()(Union(Reference(R(0)), Reference(R(1)))))
        )

    def test_difference(self):
        self.assertEqual(
            Difference(Reference(R(0)), Reference(R(1))),
            ToCsgAst()(
                ToAst()(Difference(Reference(R(0)), Reference(R(1))))
            )
        )


class TestGetSamples(unittest.TestCase):
    def test_not_reference(self):
        dataset = Dataset(1, 1, 1, 45)

        def to_action_sequence(code):
            return AstToSingleActionSequence()(ToAst()(code))

        samples = get_samples(dataset, to_action_sequence)
        self.assertEqual(7, len(samples.rules))
        self.assertEqual(12, len(samples.node_types))
        self.assertEqual(8, len(samples.tokens))

    def test_reference(self):
        dataset = Dataset(1, 1, 1, 45, reference=True)

        def to_action_sequence(code):
            return AstToSingleActionSequence()(ToAst()(code))

        samples = get_samples(dataset, to_action_sequence)
        self.assertEqual(7, len(samples.rules))
        self.assertEqual(12, len(samples.node_types))
        self.assertEqual(8, len(samples.tokens))


if __name__ == "__main__":
    unittest.main()
