import numpy as np
import torch
from torch import nn
import unittest
from mlprogram.nn import Apply
from mlprogram.nn.utils.rnn import pad_sequence


class MockModule(nn.Module):
    def __init__(self, k: int):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(k, dtype=torch.float))

    def forward(self, x):
        assert len(x.shape) == 2
        return x + self.p


class TestApply(unittest.TestCase):
    def test_parameters(self):
        apply = Apply("in", "out", MockModule(1))
        self.assertEqual(set(["module.p"]),
                         dict(apply.named_parameters()).keys())

    def test_simple(self):
        apply = Apply("in", "out", MockModule(1))
        output = apply({"in": torch.arange(3).reshape(-1, 1)})
        self.assertTrue(np.array_equal(
            [[1], [2], [3]], output["out"].detach().numpy()
        ))

    def test_sequence(self):
        apply = Apply("in", "out", MockModule(1),
                      value_type="list")

        output = apply({"in": [torch.arange(2).reshape(-1, 1),
                               torch.arange(1).reshape(-1, 1) * 10,
                               torch.arange(3).reshape(-1, 1) * 100]})
        self.assertTrue(np.array_equal(
            [[1], [2]], output["out"][0].detach().numpy()
        ))
        self.assertTrue(np.array_equal(
            [[1]], output["out"][1].detach().numpy()
        ))
        self.assertTrue(np.array_equal(
            [[1], [101], [201]], output["out"][2].detach().numpy()
        ))

    def test_empty_sequence(self):
        apply = Apply("in", "out", MockModule(1),
                      value_type="list")

        output = apply({"in": []})
        self.assertEqual([], output["out"])
        output = apply({"in": [torch.zeros((0,))]})
        self.assertEqual([[]], output["out"])

    def test_padded_sequence(self):
        apply = Apply("in", "out", MockModule(1),
                      value_type="padded_tensor")

        padded = pad_sequence([torch.arange(2).reshape(-1, 1),
                               torch.arange(1).reshape(-1, 1) * 10,
                               torch.arange(3).reshape(-1, 1) * 100],
                              padding_value=-1)
        output = apply({"in": padded})
        self.assertTrue(np.array_equal(
            [
                [[1], [1], [1]],
                [[2], [0], [101]],
                [[0], [0], [201]]
            ],
            output["out"].data.detach().numpy()
        ))
        self.assertTrue(np.array_equal(
            [
                [1, 1, 1],
                [1, 0, 1],
                [0, 0, 1]
            ],
            output["out"].mask.detach().numpy()
        ))


if __name__ == "__main__":
    unittest.main()
