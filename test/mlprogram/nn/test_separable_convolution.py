import torch

from mlprogram.nn import SeparableConv1d


class TestSeparableConv1d(object):
    def test_parameters(self):
        sconv = SeparableConv1d(2, 5, 3, padding=1, bias=False)
        pshape = {k: v.shape for k, v in sconv.named_parameters()}
        assert 2 == len(list(sconv.parameters()))
        assert (2, 1, 3) == pshape["depthwise_conv.weight"]
        assert (5, 2, 1) == pshape["pointwise_conv.weight"]

        sconv = SeparableConv1d(2, 5, 3, padding=1, bias=True)
        pshape = {k: v.shape for k, v in sconv.named_parameters()}
        assert 3 == len(list(sconv.parameters()))
        assert (2, 1, 3) == pshape["depthwise_conv.weight"]
        assert (5, 2, 1) == pshape["pointwise_conv.weight"]
        assert (5,) == pshape["pointwise_conv.bias"]

    def test_shape(self):
        sconv = SeparableConv1d(2, 5, 3, padding=1)
        output = sconv(torch.Tensor(7, 2, 11))
        assert (7, 5, 11) == output.shape
