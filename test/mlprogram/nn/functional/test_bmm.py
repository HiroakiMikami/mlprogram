import numpy as np
import torch

from mlprogram.nn.functional import bmm


class TestBmm(object):
    def test_dense(self):
        torch.manual_seed(0)
        input = torch.randn(2, 3, 5)
        output = torch.randn(2, 5, 7)
        assert np.allclose(
            torch.bmm(input, output).numpy(),
            bmm(input, output).numpy())

    def test_sparse(self):
        torch.manual_seed(0)
        input = torch.randn(2, 3, 5)
        output = torch.randn(2, 5, 7)
        assert np.allclose(
            torch.bmm(input, output).numpy(),
            bmm(input.to_sparse(), output).numpy())
