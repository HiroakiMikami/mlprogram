import unittest

from mlprogram.language.csg import is_subtype


class TestIsSubtype(unittest.TestCase):
    def test_base_type_is_CSG(self):
        self.assertTrue(is_subtype("Rectangle", "CSG"))
        self.assertTrue(is_subtype("Circle", "CSG"))
        self.assertTrue(is_subtype("Transolation", "CSG"))
        self.assertTrue(is_subtype("Rotation", "CSG"))
        self.assertTrue(is_subtype("Union", "CSG"))
        self.assertTrue(is_subtype("Difference", "CSG"))
        self.assertTrue(is_subtype("CSG", "CSG"))
        self.assertFalse(is_subtype("number", "CSG"))

    def test_base_type_is_number(self):
        self.assertTrue(is_subtype("number", "number"))
        self.assertFalse(is_subtype("CSG", "number"))


if __name__ == "__main__":
    unittest.main()
