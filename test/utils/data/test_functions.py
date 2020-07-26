import torch
import unittest
import numpy as np
import ast
from mlprogram.utils import Query, Token
from mlprogram.languages.python import to_ast
from mlprogram.utils.data \
    import ListDataset, get_samples, get_words, \
    get_characters, Collate, CollateOptions
from mlprogram.utils.transform import AstToSingleActionSequence


def tokenize_query(str: str) -> Query:
    return Query(
        list(map(lambda x: Token(None, x), str.split(" "))),
        str.split(" "))


def to_action_sequence(code: str):
    return AstToSingleActionSequence()(
        to_ast(ast.parse(code).body[0], lambda x: [x]))


class TestGetWords(unittest.TestCase):
    def test_get_words(self):
        entries = [{"input": ["foo bar"], "ground_truth": ["y = x + 1"]},
                   {"input": ["test foo"], "ground_truth": ["f(x)"]}]
        dataset = ListDataset(entries)
        words = get_words(dataset, tokenize_query)
        self.assertEqual(["foo", "bar", "test", "foo"], words)


class TestGetCharacters(unittest.TestCase):
    def test_get_characters(self):
        entries = [{"input": ["foo bar"], "ground_truth": ["y = x + 1"]},
                   {"input": ["test foo"], "ground_truth": ["f(x)"]}]
        dataset = ListDataset(entries)
        chars = get_characters(dataset, tokenize_query)
        self.assertEqual([
            "f", "o", "o",
            "b", "a", "r",
            "t", "e", "s", "t",
            "f", "o", "o"], chars)


class TestGetSamples(unittest.TestCase):
    def test_get_samples(self):
        entries = [{"input": ["foo bar"], "ground_truth": ["y = x + 1"]},
                   {"input": ["test foo"], "ground_truth": ["f(x)"]}]
        dataset = ListDataset(entries)
        d = get_samples(dataset, to_action_sequence)
        self.assertEqual(["y", "x", "1", "f", "x"], d.tokens)
        self.assertEqual(12, len(d.rules))
        self.assertEqual(28, len(d.node_types))


class TestCollate(unittest.TestCase):
    def test_collate(self):
        data = [
            {
                "pad0": torch.zeros(1),
                "pad1": torch.ones(2, 1),
                "stack0": torch.zeros(1),
                "stack1": torch.ones(1, 3)
            },
            {
                "pad0": torch.zeros(2) + 1,
                "pad1": torch.ones(1, 1) + 1,
                "stack0": torch.zeros(1) + 1,
                "stack1": torch.ones(1, 3) + 1
            }
        ]
        collate = Collate(device=torch.device("cpu"),
                          pad0=CollateOptions(True, 0, -1),
                          pad1=CollateOptions(True, 0, -1),
                          stack0=CollateOptions(False, 0, -1),
                          stack1=CollateOptions(False, 1, -1))
        retval = collate.collate(data)
        self.assertEqual(set(["pad0", "pad1", "stack0", "stack1"]),
                         set(retval.keys()))

        self.assertTrue(np.array_equal([[0, 1], [-1, 1]],
                                       retval["pad0"].data.numpy()))
        self.assertTrue(np.array_equal([[1, 1], [0, 1]],
                                       retval["pad0"].mask.numpy()))
        self.assertTrue(np.array_equal([[[1], [2]], [[1], [-1]]],
                                       retval["pad1"].data.numpy()))
        self.assertTrue(np.array_equal([[1, 1], [1, 0]],
                                       retval["pad1"].mask.numpy()))

        self.assertTrue(np.array_equal([[0], [1]],
                                       retval["stack0"].data.numpy()))
        self.assertTrue(np.array_equal([[[1, 1, 1], [2, 2, 2]]],
                                       retval["stack1"].data.numpy()))

    def test_collate_with_skip(self):
        data = [
            {
                "pad0": torch.zeros(1),
            },
            None
        ]
        collate = Collate(device=torch.device("cpu"),
                          pad0=CollateOptions(True, 0, -1))
        retval = collate.collate(data)
        self.assertEqual(set(["pad0"]),
                         set(retval.keys()))
        self.assertTrue(np.array_equal([[0]],
                                       retval["pad0"].data.numpy()))

    def test_collate_with_additional_key(self):
        data = [
            {"pad0": 1},
            {"pad0": 2}
        ]
        collate = Collate(device=torch.device("cpu"))
        retval = collate.collate(data)
        self.assertEqual(set(["pad0"]),
                         set(retval.keys()))
        self.assertEqual([1, 2], retval["pad0"])

    def test_collate_with_all_none_batch(self):
        data = [None]
        collate = Collate(device=torch.device("cpu"),
                          pad0=CollateOptions(True, 0, -1))
        retval = collate.collate(data)
        self.assertEqual({}, retval)

    def test_collate_with_pad(self):
        data = [
            {
                "x": torch.zeros(2, 1),
            },
            {
                "x": torch.zeros(1, 2),
            }
        ]
        collate = Collate(device=torch.device("cpu"),
                          x=CollateOptions(False, 0, -1))
        retval = collate.collate(data)
        self.assertEqual(set(["x"]), set(retval.keys()))
        self.assertTrue(np.array_equal((2, 2, 2),
                                       retval["x"].shape))
        self.assertTrue(np.array_equal([[[0, -1], [0, -1]],
                                        [[0, 0], [-1, -1]]],
                                       retval["x"].numpy()))

    def test_split(self):
        data = [
            {
                "pad0": torch.zeros(1),
                "pad1": torch.ones(2, 1),
                "stack0": torch.zeros(1),
                "stack1": torch.ones(1, 3)
            },
            {
                "pad0": torch.zeros(2) + 1,
                "pad1": torch.ones(1, 1) + 1,
                "stack0": torch.zeros(1) + 1,
                "stack1": torch.ones(1, 3) + 1
            }
        ]
        collate = Collate(device=torch.device("cpu"),
                          pad0=CollateOptions(True, 0, -1),
                          pad1=CollateOptions(True, 0, -1),
                          stack0=CollateOptions(False, 0, -1),
                          stack1=CollateOptions(False, 1, -1))
        retval = collate.split(collate.collate(data))
        self.assertEqual(2, len(retval))
        for i in range(2):
            expected = data[i]
            actual = retval[i]
            self.assertEqual(set(expected.keys()), set(actual.keys()))
            for key in expected:
                self.assertTrue(np.array_equal(expected[key], actual[key]))

    def test_split_with_additional_key(self):
        data = [
            {"pad0": 1},
            {"pad0": 2}
        ]
        collate = Collate(device=torch.device("cpu"))
        retval = collate.split(collate.collate(data))
        self.assertEqual(1, retval[0]["pad0"])
        self.assertEqual(2, retval[1]["pad0"])


if __name__ == "__main__":
    unittest.main()
