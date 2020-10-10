import torch
from mlprogram.utils.data import random_split


class TestRandomSplit(object):
    def test_happy_path(self):
        dataset = torch.utils.data.TensorDataset(torch.rand(10,))
        datasets = random_split(dataset, {"0": 0.5, "1": 0.5}, 0)
        assert 2 == len(datasets)
        assert 5 == len(datasets["0"])
        assert 5 == len(datasets["1"])

    def test_reamain(self):
        dataset = torch.utils.data.TensorDataset(torch.rand(11,))
        datasets = random_split(dataset, {"0": 0.5, "1": 0.5}, 0)
        assert 2 == len(datasets)
        assert 6 == len(datasets["0"])
        assert 5 == len(datasets["1"])
