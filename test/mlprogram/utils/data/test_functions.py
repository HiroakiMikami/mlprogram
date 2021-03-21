from typing import List

import numpy as np
import torch

from mlprogram.builtins import Environment
from mlprogram.languages import Analyzer, Token
from mlprogram.languages.python import Parser
from mlprogram.utils.data import (
    Collate,
    CollateOptions,
    ListDataset,
    get_characters,
    get_samples,
    get_words,
    split_by_n_error,
)


def tokenize(str: str) -> List[Token]:
    return list(map(lambda x: Token(None, x, x), str.split(" ")))


class TestGetWords(object):
    def test_get_words(self):
        entries = [Environment({"text_query": "foo bar", "ground_truth": "y = x + 1"},
                               set(["ground_truth"])),
                   Environment({"text_query": "test foo", "ground_truth": "f(x)"},
                               set(["ground_truth"]))]
        dataset = ListDataset(entries)
        words = get_words(dataset, tokenize)
        assert ["foo", "bar", "test", "foo"] == words


class TestGetCharacters(object):
    def test_get_characters(self):
        entries = [Environment({"text_query": "foo bar", "ground_truth": "y = x + 1"},
                               set(["ground_truth"])),
                   Environment({"text_query": "test foo", "ground_truth": "f(x)"},
                               set(["ground_truth"]))]
        dataset = ListDataset(entries)
        chars = get_characters(dataset, tokenize)
        assert [
            "f", "o", "o",
            "b", "a", "r",
            "t", "e", "s", "t",
            "f", "o", "o"] == chars


class TestGetSamples(object):
    def test_get_samples(self):
        entries = [Environment({"ground_truth": "y = x + 1"}, set(["ground_truth"])),
                   Environment({"ground_truth": "f(x)"}, set(["ground_truth"]))]
        dataset = ListDataset(entries)
        d = get_samples(dataset, Parser(lambda x: [x]))
        assert [
            ("str", "y"),
            ("str", "x"),
            ("int", "1"),
            ("str", "f"),
            ("str", "x")
        ] == d.tokens
        assert 12 == len(d.rules)
        assert 28 == len(d.node_types)


class TestCollate(object):
    def test_collate(self):
        data = [
            Environment({
                "pad0": torch.zeros(1),
                "pad1": torch.ones(2, 1),
                "stack0": torch.zeros(1),
                "stack1": torch.ones(1, 3)
            }),
            Environment({
                "pad0": torch.zeros(2) + 1,
                "pad1": torch.ones(1, 1) + 1,
                "stack0": torch.zeros(1) + 1,
                "stack1": torch.ones(1, 3) + 1
            })
        ]
        collate = Collate(pad0=CollateOptions(True, 0, -1),
                          pad1=CollateOptions(True, 0, -1),
                          stack0=CollateOptions(False, 0, -1),
                          stack1=CollateOptions(False, 1, -1))
        retval = collate.collate(data)
        assert set(retval.to_dict().keys()) == set([
            "pad0", "pad1", "stack0", "stack1"
        ])

        assert np.array_equal([[0, 1], [-1, 1]],
                              retval["pad0"].data.numpy())
        assert np.array_equal([[1, 1], [0, 1]],
                              retval["pad0"].mask.numpy())
        assert np.array_equal([[[1], [2]], [[1], [-1]]],
                              retval["pad1"].data.numpy())
        assert np.array_equal([[1, 1], [1, 0]],
                              retval["pad1"].mask.numpy())

        assert np.array_equal([[0], [1]],
                              retval["stack0"].data.numpy())
        assert np.array_equal([[[1, 1, 1], [2, 2, 2]]],
                              retval["stack1"].data.numpy())

    def test_collate_with_skip(self):
        data = [
            Environment({
                "pad0": torch.zeros(1),
            }),
            None
        ]
        collate = Collate(pad0=CollateOptions(True, 0, -1))
        retval = collate.collate(data)
        assert set(["pad0"]) == set(retval.to_dict().keys())
        assert np.array_equal([[0]], retval["pad0"].data.numpy())

    def test_collate_with_additional_key(self):
        data = [
            Environment({"pad0": 1}),
            Environment({"pad0": 2})
        ]
        collate = Collate()
        retval = collate.collate(data)
        assert set(["pad0"]) == set(retval.to_dict().keys())
        assert [1, 2] == retval["pad0"]

    def test_collate_with_all_none_batch(self):
        data = [None]
        collate = Collate(pad0=CollateOptions(True, 0, -1))
        retval = collate.collate(data)
        assert {} == retval.to_dict()

    def test_collate_with_pad(self):
        data = [
            Environment({"x": torch.zeros(2, 1)}),
            Environment({"x": torch.zeros(1, 2)})
        ]
        collate = Collate(x=CollateOptions(False, 0, -1))
        retval = collate.collate(data)
        assert set(["x"]) == set(retval.to_dict().keys())
        assert np.array_equal((2, 2, 2), retval["x"].shape)
        assert np.array_equal([[[0, -1], [0, -1]],
                               [[0, 0], [-1, -1]]],
                              retval["x"].numpy())

    def test_split(self):
        data = [
            Environment({
                "pad0": torch.zeros(1),
                "pad1": torch.ones(2, 1),
                "stack0": torch.zeros(1),
                "stack1": torch.ones(1, 3)
            }),
            Environment({
                "pad0": torch.zeros(2) + 1,
                "pad1": torch.ones(1, 1) + 1,
                "stack0": torch.zeros(1) + 1,
                "stack1": torch.ones(1, 3) + 1
            })
        ]
        collate = Collate(pad0=CollateOptions(True, 0, -1),
                          pad1=CollateOptions(True, 0, -1),
                          stack0=CollateOptions(False, 0, -1),
                          stack1=CollateOptions(False, 1, -1))
        retval = collate.split(collate.collate(data))
        assert 2 == len(retval)
        for i in range(2):
            expected = data[i]
            actual = retval[i]
            assert set(expected.to_dict().keys()) == \
                set(actual.to_dict().keys())
            for key in expected.to_dict():
                assert np.array_equal(expected[key], actual[key])

    def test_split_with_additional_key(self):
        data = [
            Environment({"pad0": 1}),
            Environment({"pad0": 2})
        ]
        collate = Collate()
        retval = collate.split(collate.collate(data))
        assert 1 == retval[0]["pad0"]
        assert 2 == retval[1]["pad0"]


class MockAnalyzer(Analyzer[str, str]):
    def __init__(self, errors):
        self.errors = errors

    def __call__(self, code):
        return self.errors[code]


class TestSplitByNError(object):
    def test_split(self):
        dataset = ListDataset([
            Environment({"code": "x"}),
            Environment({"code": "y"})
        ])
        splitted = split_by_n_error(dataset,
                                    MockAnalyzer({
                                        "x": [],
                                        "y": ["error"]
                                    }))
        assert list(splitted["no_error"]) == [
            Environment({"code": "x"})
        ]
        assert list(splitted["with_error"]) == [
            Environment({"code": "y"})
        ]

    def test_precomputed_n_error(self):
        dataset = ListDataset([
            Environment({"code": "x", "n_error": 0}, set(["n_error"])),
            Environment({"code": "y"})
        ])
        splitted = split_by_n_error(dataset,
                                    MockAnalyzer({
                                        "x": ["error"],
                                        "y": ["error"]
                                    }))
        assert list(splitted["no_error"]) == [
            Environment({"code": "x", "n_error": 0}, set(["n_error"]))
        ]
        assert list(splitted["with_error"]) == [
            Environment({"code": "y"})
        ]

    def test_multiprocess(self):
        dataset = ListDataset([
            Environment({"code": "x"}),
            Environment({"code": "y"})
        ])
        splitted = split_by_n_error(dataset,
                                    MockAnalyzer({
                                        "x": [],
                                        "y": ["error"]
                                    }), n_process=2)
        assert list(splitted["no_error"]) == [
            Environment({"code": "x"})
        ]
        assert list(splitted["with_error"]) == [
            Environment({"code": "y"})
        ]
