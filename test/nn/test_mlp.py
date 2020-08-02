import unittest
import torch
import numpy as np

from mlprogram.nn import MLP
from torch import nn


class TestMLP(unittest.TestCase):
    def test_parameters(self):
        mlp = MLP(1, 2, 3, 1, 2)
        params = dict(mlp.named_parameters())
        self.assertEqual(4, len(params))
        self.assertEqual((3, 1),
                         params["module.block0.linear0.weight"].shape)
        self.assertEqual((3,), params["module.block0.linear0.bias"].shape)
        self.assertEqual((2, 3),
                         params["module.block1.linear0.weight"].shape)
        self.assertEqual((2,), params["module.block1.linear0.bias"].shape)

    def test_shape(self):
        mlp = MLP(1, 2, 3, 1, 2)
        out = mlp(torch.rand(1, 1))
        self.assertEqual((1, 2), out.shape)

    def test_activation(self):
        mlp = MLP(1, 2, 3, 1, 2, activation=nn.Sigmoid())
        out = mlp(torch.rand(1, 1))
        self.assertEqual((1, 2), out.shape)
        self.assertTrue(torch.all(0 <= out))
        self.assertTrue(torch.all(out <= 1))

    def test_value(self):
        mlp = MLP(1, 2, 3, 1, 2, activation=nn.Sigmoid())
        input = torch.zeros(2, 1)
        input[0] = 1
        out = mlp(input)
        self.assertEqual((2, 2), out.shape)
        self.assertFalse(np.array_equal(out[0, :].detach().numpy(),
                                        out[1, :].detach().numpy()))


if __name__ == "__main__":
    unittest.main()
