import torch
import unittest
import numpy as np
import ast
from mlprogram.utils import Query
from mlprogram.languages.python import to_ast
from mlprogram.actions import ActionOptions
from mlprogram.utils.data \
    import Entry, ListDataset, to_eval_dataset, get_samples, get_words, \
    get_characters, Collate, CollateOptions
from mlprogram.utils.transform import AstToSingleActionSequence


def tokenize(query: str):
    return query.split(" ")


def tokenize_query(query: str):
    return Query(query.split(" "), query.split(" "))


def to_action_sequence(code: str):
    return AstToSingleActionSequence(tokenize=tokenize)(
        to_ast(ast.parse(code).body[0]))


class TestGetWords(unittest.TestCase):
    def test_get_words(self):
        entries = [Entry("foo bar", "y = x + 1"),
                   Entry("test foo", "f(x)")]
        dataset = ListDataset([entries])
        words = get_words(dataset, tokenize_query)
        self.assertEqual(["foo", "bar", "test", "foo"], words)


class TestGetCharacters(unittest.TestCase):
    def test_get_characters(self):
        entries = [Entry("foo bar", "y = x + 1"),
                   Entry("test foo", "f(x)")]
        dataset = ListDataset([entries])
        chars = get_characters(dataset, tokenize_query)
        self.assertEqual([
            "f", "o", "o",
            "b", "a", "r",
            "t", "e", "s", "t",
            "f", "o", "o"], chars)


class TestGetSamples(unittest.TestCase):
    def test_get_samples(self):
        entries = [Entry("foo bar", "y = x + 1"),
                   Entry("test foo", "f(x)")]
        dataset = ListDataset([entries])
        d = get_samples(dataset, tokenize, to_action_sequence)
        self.assertEqual(["y", "x", "1", "f", "x"], d.tokens)
        self.assertEqual(12, len(d.rules))
        self.assertEqual(28, len(d.node_types))
        self.assertEqual(ActionOptions(True, True), d.options)


class TestToEvalDataset(unittest.TestCase):
    def test_simple_case(self):
        groups = [[Entry("foo bar", "y = x1")]]
        dataset = ListDataset(groups)
        vdataset = to_eval_dataset(dataset)
        query, ref = vdataset[0]
        self.assertEqual("foo bar", query)
        self.assertEqual(["y = x1"], ref)


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


if __name__ == "__main__":
    unittest.main()
