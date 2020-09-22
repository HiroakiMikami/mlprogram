import unittest
import torch
from mlprogram.utils.data import random_split


class TestRandomSplit(unittest.TestCase):
    def test_happy_path(self):
        dataset = torch.utils.data.TensorDataset(torch.rand(10,))
        datasets = random_split(dataset, {"0": 0.5, "1": 0.5}, 0)
        self.assertEqual(2, len(datasets))
        self.assertEqual(5, len(datasets["0"]))
        self.assertEqual(5, len(datasets["1"]))

    def test_reamain(self):
        dataset = torch.utils.data.TensorDataset(torch.rand(11,))
        datasets = random_split(dataset, {"0": 0.5, "1": 0.5}, 0)
        self.assertEqual(2, len(datasets))
        self.assertEqual(6, len(datasets["0"]))
        self.assertEqual(5, len(datasets["1"]))


if __name__ == "__main__":
    unittest.main()
