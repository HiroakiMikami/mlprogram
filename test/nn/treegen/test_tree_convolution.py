import unittest
import torch
import numpy as np

from nl2prog.nn.treegen import TreeConvolution


class TestTreeConvolution(unittest.TestCase):
    def test_parameters(self):
        tconv = TreeConvolution(2, 5, 3, bias=False)
        pshape = {k: v.shape for k, v in tconv.named_parameters()}
        self.assertEqual(1, len(list(tconv.parameters())))
        self.assertEqual((5, 6, 1), pshape["conv.weight"])

        tconv = TreeConvolution(2, 5, 3, bias=True)
        pshape = {k: v.shape for k, v in tconv.named_parameters()}
        self.assertEqual(2, len(list(tconv.parameters())))
        self.assertEqual((5, 6, 1), pshape["conv.weight"])
        self.assertEqual((5,), pshape["conv.bias"])

    def test_shape(self):
        tconv = TreeConvolution(2, 5, 3)
        output = tconv(torch.Tensor(7, 2, 11), torch.Tensor(7, 11, 11))
        self.assertEqual((7, 5, 11), output.shape)

    def test_dense_matrix(self):
        """
        0 -> 1
        """
        input = torch.tensor([[[1, 2]]]).float()
        m = torch.tensor([[0, 1], [0, 0]]).float().view(1, 2, 2)

        tconv = TreeConvolution(1, 1, 1, bias=False)
        torch.nn.init.ones_(tconv.conv.weight)
        output = tconv(input, m)
        self.assertTrue(np.allclose([[[1, 2]]], output.detach().numpy()))

        tconv = TreeConvolution(1, 2, 2, bias=False)
        torch.nn.init.eye_(tconv.conv.weight.view(2, 2))
        """
        input   = [1, 2]
        input*M = [0, 1]
        """
        output = tconv(input, m)
        self.assertTrue(np.allclose(
            [[[1, 2], [0, 1]]],
            output.detach().numpy()))


if __name__ == "__main__":
    unittest.main()
