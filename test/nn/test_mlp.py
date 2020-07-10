import unittest
import torch

from mlprogram.nn import MLP


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


if __name__ == "__main__":
    unittest.main()
