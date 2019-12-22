import unittest
import torch
import numpy as np

from nl2prog.nn.functional import bmm


class TestBmm(unittest.TestCase):
    def test_dense(self):
        input = torch.randn(2, 3, 5)
        output = torch.randn(2, 5, 7)
        self.assertTrue(np.allclose(
            torch.bmm(input, output).numpy(),
            bmm(input, output).numpy()))

    def test_sparse(self):
        input = torch.randn(2, 3, 5)
        output = torch.randn(2, 5, 7)
        self.assertTrue(np.allclose(
            torch.bmm(input, output).numpy(),
            bmm(input.to_sparse(), output).numpy()))


if __name__ == "__main__":
    unittest.main()
