import unittest
from mlprogram.languages.csg import IsSubtype
from mlprogram.languages.csg import Parser
from mlprogram.languages.csg import get_samples, Dataset


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

        samples = get_samples(dataset, Parser())
        self.assertEqual(7, len(samples.rules))
        self.assertEqual(12, len(samples.node_types))
        self.assertEqual(9, len(samples.tokens))

    def test_reference(self):
        dataset = Dataset(1, 1, 1, 1, 45, reference=True)

        samples = get_samples(dataset, Parser())
        self.assertEqual(7, len(samples.rules))
        self.assertEqual(12, len(samples.node_types))
        self.assertEqual(9, len(samples.tokens))


if __name__ == "__main__":
    unittest.main()
