import unittest
from mlprogram.interpreters import Reference as R
from mlprogram.actions import AstToActionSequence
from mlprogram.languages.csg import GetTokenType, IsSubtype
from mlprogram.languages.csg import Parser
from mlprogram.languages.csg import get_samples, Dataset


class TestGetTokenType(unittest.TestCase):
    def test_csg_type(self):
        self.assertEqual("CSG", GetTokenType()(R("0")))

    def test_number(self):
        self.assertEqual("int", GetTokenType()(0))


class TestIsSubType(unittest.TestCase):
    def test_happy_path(self):
        self.assertTrue(IsSubtype()("CSG", "CSG"))
        self.assertTrue(IsSubtype()("Circle", "CSG"))

    def test_integer(self):
        self.assertTrue(IsSubtype()("int", "size"))
        self.assertFalse(IsSubtype()("int", "CSG"))


class TestGetSamples(unittest.TestCase):
    def test_not_reference(self):
        dataset = Dataset(1, 1, 1, 1, 45)

        def to_action_sequence(code):
            return AstToActionSequence()(Parser().parse(code))

        samples = get_samples(dataset, to_action_sequence)
        self.assertEqual(7, len(samples.rules))
        self.assertEqual(12, len(samples.node_types))
        self.assertEqual(8, len(samples.tokens))

    def test_reference(self):
        dataset = Dataset(1, 1, 1, 1, 45, reference=True)

        def to_action_sequence(code):
            return AstToActionSequence()(Parser().parse(code))

        samples = get_samples(dataset, to_action_sequence)
        self.assertEqual(7, len(samples.rules))
        self.assertEqual(12, len(samples.node_types))
        self.assertEqual(8, len(samples.tokens))


if __name__ == "__main__":
    unittest.main()
