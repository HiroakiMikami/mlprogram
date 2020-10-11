import torch
import numpy as np
from typing import List
from mlprogram import Environment
from mlprogram.languages import Token
from mlprogram.languages.python import Parser
from mlprogram.utils.data \
    import ListDataset, get_samples, get_words, \
    get_characters, Collate, CollateOptions


def tokenize(str: str) -> List[Token]:
    return list(map(lambda x: Token(None, x, x), str.split(" ")))


class TestGetWords(object):
    def test_get_words(self):
        entries = [Environment(inputs={"input": "foo bar"},
                               supervisions={"ground_truth": "y = x + 1"}),
                   Environment(inputs={"input": "test foo"},
                               supervisions={"ground_truth": "f(x)"})]
        dataset = ListDataset(entries)
        words = get_words(dataset, tokenize)
        assert ["foo", "bar", "test", "foo"] == words


class TestGetCharacters(object):
    def test_get_characters(self):
        entries = [Environment(inputs={"input": "foo bar"},
                               supervisions={"ground_truth": "y = x + 1"}),
                   Environment(inputs={"input": "test foo"},
                               supervisions={"ground_truth": "f(x)"})]
        dataset = ListDataset(entries)
        chars = get_characters(dataset, tokenize)
        assert [
            "f", "o", "o",
            "b", "a", "r",
            "t", "e", "s", "t",
            "f", "o", "o"] == chars


class TestGetSamples(object):
    def test_get_samples(self):
        entries = [Environment(inputs={"input": "foo bar"},
                               supervisions={"ground_truth": "y = x + 1"}),
                   Environment(inputs={"input": "test foo"},
                               supervisions={"ground_truth": "f(x)"})]
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
            Environment(inputs={
                "pad0": torch.zeros(1),
                "pad1": torch.ones(2, 1),
                "stack0": torch.zeros(1),
                "stack1": torch.ones(1, 3)
            }),
            Environment(inputs={
                "pad0": torch.zeros(2) + 1,
                "pad1": torch.ones(1, 1) + 1,
                "stack0": torch.zeros(1) + 1,
                "stack1": torch.ones(1, 3) + 1
            })
        ]
        collate = Collate(device=torch.device("cpu"),
                          pad0=CollateOptions(True, 0, -1),
                          pad1=CollateOptions(True, 0, -1),
                          stack0=CollateOptions(False, 0, -1),
                          stack1=CollateOptions(False, 1, -1))
        retval = collate.collate(data)
        assert set(retval.to_dict().keys()) == set([
            "input@pad0", "input@pad1", "input@stack0", "input@stack1"
        ])

        assert np.array_equal([[0, 1], [-1, 1]],
                              retval.inputs["pad0"].data.numpy())
        assert np.array_equal([[1, 1], [0, 1]],
                              retval.inputs["pad0"].mask.numpy())
        assert np.array_equal([[[1], [2]], [[1], [-1]]],
                              retval.inputs["pad1"].data.numpy())
        assert np.array_equal([[1, 1], [1, 0]],
                              retval.inputs["pad1"].mask.numpy())

        assert np.array_equal([[0], [1]],
                              retval.inputs["stack0"].data.numpy())
        assert np.array_equal([[[1, 1, 1], [2, 2, 2]]],
                              retval.inputs["stack1"].data.numpy())

    def test_collate_with_skip(self):
        data = [
            Environment(inputs={
                "pad0": torch.zeros(1),
            }),
            None
        ]
        collate = Collate(device=torch.device("cpu"),
                          pad0=CollateOptions(True, 0, -1))
        retval = collate.collate(data)
        assert set(["input@pad0"]) == set(retval.to_dict().keys())
        assert np.array_equal([[0]], retval["input@pad0"].data.numpy())

    def test_collate_with_additional_key(self):
        data = [
            Environment(inputs={"pad0": 1}),
            Environment(inputs={"pad0": 2})
        ]
        collate = Collate(device=torch.device("cpu"))
        retval = collate.collate(data)
        assert set(["input@pad0"]) == set(retval.to_dict().keys())
        assert [1, 2] == retval["input@pad0"]

    def test_collate_with_all_none_batch(self):
        data = [None]
        collate = Collate(device=torch.device("cpu"),
                          pad0=CollateOptions(True, 0, -1))
        retval = collate.collate(data)
        assert {} == retval.to_dict()

    def test_collate_with_pad(self):
        data = [
            Environment(inputs={
                "x": torch.zeros(2, 1),
            }),
            Environment(inputs={
                "x": torch.zeros(1, 2),
            })
        ]
        collate = Collate(device=torch.device("cpu"),
                          x=CollateOptions(False, 0, -1))
        retval = collate.collate(data)
        assert set(["input@x"]) == set(retval.to_dict().keys())
        assert np.array_equal((2, 2, 2), retval["input@x"].shape)
        assert np.array_equal([[[0, -1], [0, -1]],
                               [[0, 0], [-1, -1]]],
                              retval["input@x"].numpy())

    def test_split(self):
        data = [
            Environment(inputs={
                "pad0": torch.zeros(1),
                "pad1": torch.ones(2, 1),
                "stack0": torch.zeros(1),
                "stack1": torch.ones(1, 3)
            }),
            Environment(inputs={
                "pad0": torch.zeros(2) + 1,
                "pad1": torch.ones(1, 1) + 1,
                "stack0": torch.zeros(1) + 1,
                "stack1": torch.ones(1, 3) + 1
            })
        ]
        collate = Collate(device=torch.device("cpu"),
                          pad0=CollateOptions(True, 0, -1),
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
            Environment(inputs={"pad0": 1}),
            Environment(inputs={"pad0": 2})
        ]
        collate = Collate(device=torch.device("cpu"))
        retval = collate.split(collate.collate(data))
        assert 1 == retval[0]["input@pad0"]
        assert 2 == retval[1]["input@pad0"]
