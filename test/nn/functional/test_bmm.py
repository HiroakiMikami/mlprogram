import unittest
import torch
import numpy as np

from mlprogram.nn.functional import bmm


class TestBmm(unittest.TestCase):
    def test_dense(self):
        torch.manual_seed(0)
        input = torch.randn(2, 3, 5)
        output = torch.randn(2, 5, 7)
        self.assertTrue(np.allclose(
            torch.bmm(input, output).numpy(),
            bmm(input, output).numpy()))

    def test_sparse(self):
        torch.manual_seed(0)
        input = torch.randn(2, 3, 5)
        output = torch.randn(2, 5, 7)
        self.assertTrue(np.allclose(
            torch.bmm(input, output).numpy(),
            bmm(input.to_sparse(), output).numpy()))


if __name__ == "__main__":
    unittest.main()
