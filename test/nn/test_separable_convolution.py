import unittest
import torch

from nl2prog.nn import SeparableConv1d


class TestSeparableConv1d(unittest.TestCase):
    def test_parameters(self):
        sconv = SeparableConv1d(2, 5, 3, padding=1, bias=False)
        pshape = {k: v.shape for k, v in sconv.named_parameters()}
        self.assertEqual(2, len(list(sconv.parameters())))
        self.assertEqual((2, 1, 3), pshape["depthwise_conv.weight"])
        self.assertEqual((5, 2, 1), pshape["pointwise_conv.weight"])

        sconv = SeparableConv1d(2, 5, 3, padding=1, bias=True)
        pshape = {k: v.shape for k, v in sconv.named_parameters()}
        self.assertEqual(3, len(list(sconv.parameters())))
        self.assertEqual((2, 1, 3), pshape["depthwise_conv.weight"])
        self.assertEqual((5, 2, 1), pshape["pointwise_conv.weight"])
        self.assertEqual((5,), pshape["pointwise_conv.bias"])

    def test_shape(self):
        sconv = SeparableConv1d(2, 5, 3, padding=1)
        output = sconv(torch.Tensor(7, 2, 11))
        self.assertEqual((7, 5, 11), output.shape)


if __name__ == "__main__":
    unittest.main()
