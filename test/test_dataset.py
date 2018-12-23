import unittest
from src.dataset import Dataset, Sample
from src.grammar import NodeType, Node, Rule, ROOT
import os
import numpy as np
from src.grammar import Grammar, CLOSE_NODE


class TestDataset(unittest.TestCase):
    def test_words(self):
        data = Dataset(os.path.join("test", "test_dataset"))
        self.assertEqual(set(data.words(0)), set(["foo", "bar", "test"]))
        self.assertEqual(set(data.words(2)), set())

    def test_tokens(self):
        data = Dataset(os.path.join("test", "test_dataset"))
        self.assertEqual(set(data.tokens(0)), set(["threading", "foo", "0"]))
        self.assertEqual(set(data.tokens(10)), set())

    def test_rules(self):
        data = Dataset(os.path.join("test", "test_dataset"))
        expected = set([
            Rule(ROOT, (Node("-", NodeType("ImportFrom", False)), )),
            Rule(
                NodeType("ImportFrom", False),
                (Node("module", NodeType("str", False)),
                 Node("names", NodeType("alias", True)),
                 Node("level", NodeType("int", False)))),
            Rule(
                NodeType("alias", True),
                (Node("val0", NodeType("alias", False)), )),
            Rule(
                NodeType("alias", False),
                (Node("-", NodeType("alias", False)), )),
            Rule(
                NodeType("alias", False),
                (Node("name", NodeType("str", False)), ))
        ])
        self.assertEqual(set(data.rules), expected)

    def test_node_types(self):
        data = Dataset(os.path.join("test", "test_dataset"))
        expected = set([
            ROOT,
            NodeType("ImportFrom", False),
            NodeType("str", False),
            NodeType("alias", True),
            NodeType("int", False),
            NodeType("alias", False),
            NodeType("str", False)
        ])
        self.assertEqual(set(data.node_types), expected)

    def test_next(self):
        data = Dataset(os.path.join("test", "test_dataset"))
        grammar = Grammar(data.node_types, data.rules,
                          data.tokens(0) + [CLOSE_NODE])
        data.prepare({"foo": 1, "bar": 2, "<unknown>": 0}, grammar)
        d = data.next()
        self.assertEqual(d.annotation.query, ["foo", "bar"])
        self.assertEqual(d.annotation.mappings, {})
        self.assertTrue(len(d.sequence) != 0)

        d = data.next()
        self.assertEqual(d.annotation.query, ["test"])
        self.assertEqual(d.annotation.mappings, {"foo": "bar"})

        d = data.next()
        self.assertEqual(d.annotation.query, ["foo", "bar"])
        self.assertEqual(d.annotation.mappings, {})

    def test_shuffle(self):
        data = Dataset(
            os.path.join("test", "test_dataset"),
            shuffle=True,
            rng=np.random.RandomState(0))
        grammar = Grammar(data.node_types, data.rules,
                          data.tokens(0) + [CLOSE_NODE])
        data.prepare({"foo": 1, "bar": 2, "<unknown>": 0}, grammar)
        d = data.next()
        self.assertEqual(d.annotation.query, ["test"])
        self.assertEqual(d.annotation.mappings, {"foo": "bar"})


if __name__ == "__main__":
    unittest.main()
