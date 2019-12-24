import unittest

from nl2prog.language.nl2code.action import NodeType
from nl2prog.language.bash import is_subtype


class TestIsSubtype(unittest.TestCase):
    def test_ast(self):
        self.assertTrue(is_subtype(
            NodeType("Assign", None),
            NodeType("Node", None)
        ))
        self.assertFalse(is_subtype(
            NodeType("str", None),
            NodeType("Node", None)
        ))

    def test_builtin(self):
        self.assertTrue(is_subtype(
            NodeType("str", None),
            NodeType("str", None)
        ))
        self.assertFalse(is_subtype(
            NodeType("Node", None),
            NodeType("str", None)
        ))


if __name__ == "__main__":
    unittest.main()
