import unittest

from mlprogram.languages.bash import IsSubtype


class TestIsSubtype(unittest.TestCase):
    def test_ast(self):
        self.assertTrue(IsSubtype()("Assign", "Node"))
        self.assertFalse(IsSubtype()("str", "Node"))

    def test_builtin(self):
        self.assertTrue(IsSubtype()("str", "str"))
        self.assertFalse(IsSubtype()("Node", "str"))


if __name__ == "__main__":
    unittest.main()
