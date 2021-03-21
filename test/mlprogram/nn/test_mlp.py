import numpy as np
import torch
from torch import nn

from mlprogram.nn import MLP


class TestMLP(object):
    def test_parameters(self):
        mlp = MLP(1, 2, 3, 2)
        params = dict(mlp.named_parameters())
        assert 4 == len(params)
        assert (3, 1) == params["module.linear0.weight"].shape
        assert (3,) == params["module.linear0.bias"].shape
        assert (2, 3) == params["module.linear1.weight"].shape
        assert (2,) == params["module.linear1.bias"].shape

    def test_shape(self):
        mlp = MLP(1, 2, 3, 2)
        out = mlp(torch.rand(1, 1))
        assert (1, 2) == out.shape

    def test_activation(self):
        mlp = MLP(1, 2, 3, 2, activation=nn.Sigmoid())
        out = mlp(torch.rand(1, 1))
        assert (1, 2) == out.shape
        assert torch.all(0 <= out)
        assert torch.all(out <= 1)

    def test_value(self):
        torch.manual_seed(0)
        mlp = MLP(1, 2, 3, 2, activation=nn.Sigmoid())
        input = torch.zeros(2, 1)
        input[0] = 1
        out = mlp(input)
        assert (2, 2) == out.shape
        assert not np.array_equal(out[0, :].detach().numpy(),
                                  out[1, :].detach().numpy())
