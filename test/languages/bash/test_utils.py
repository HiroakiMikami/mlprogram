import unittest

from mlprogram.languages.bash import is_subtype


class TestIsSubtype(unittest.TestCase):
    def test_ast(self):
        self.assertTrue(is_subtype("Assign", "Node"))
        self.assertFalse(is_subtype("str", "Node"))

    def test_builtin(self):
        self.assertTrue(is_subtype("str", "str"))
        self.assertFalse(is_subtype("Node", "str"))


if __name__ == "__main__":
    unittest.main()
