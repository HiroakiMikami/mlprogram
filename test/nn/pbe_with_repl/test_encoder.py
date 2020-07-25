import numpy as np
import torch
from torch import nn
import unittest
from mlprogram.nn.pbe_with_repl import Encoder


class MockModule(nn.Module):
    def __init__(self, k: int):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(k, dtype=torch.float))

    def forward(self, x):
        assert len(x.shape) == 2
        return x + self.p


class TestEncoder(unittest.TestCase):
    def test_parameters(self):
        encoder = Encoder(MockModule(1))
        self.assertEqual(set(["module.p"]),
                         dict(encoder.named_parameters()).keys())

    def test_shape(self):
        encoder = Encoder(MockModule(1))
        output = encoder({
            "input": torch.arange(2).reshape(2, 1),
            "variables": [torch.arange(3).reshape(3, 1),
                          torch.arange(1).reshape(1, 1)]
        })
        self.assertEqual((2, 2), output["input_feature"].shape)
        self.assertEqual((3, 2, 1), output["reference_features"].data.shape)
        self.assertTrue(np.array_equal(
            [[1, 1], [1, 0], [1, 0]],
            output["reference_features"].mask.numpy()
        ))

    def test_empty_sequence(self):
        encoder = Encoder(MockModule(1))
        output = encoder({
            "input": torch.arange(1).reshape(1, 1),
            "variables": [torch.arange(0).reshape(0, 1)]
        })
        self.assertEqual((1, 2), output["input_feature"].shape)
        self.assertEqual((0, 1, 1), output["reference_features"].data.shape)


if __name__ == "__main__":
    unittest.main()
