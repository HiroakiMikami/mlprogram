import torch

from mlprogram.nn import CNN2d


class TestCNN2d(object):
    def test_parameters(self):
        cnn = CNN2d(1, 2, 3, 1, 2, 2)
        params = dict(cnn.named_parameters())
        assert 4, len(params)
        assert (3, 1, 3, 3) == params["module.block0.conv0.weight"].shape
        assert (3,) == params["module.block0.conv0.bias"].shape
        assert (2, 3, 3, 3) == params["module.block1.conv0.weight"].shape
        assert (2,) == params["module.block1.conv0.bias"].shape

    def test_shape(self):
        cnn = CNN2d(1, 2, 3, 1, 2, 2)
        out = cnn(torch.rand(1, 1, 8, 8))
        assert (1, 32) == out.shape

    def test_shape_nonflatten(self):
        cnn = CNN2d(1, 2, 3, 1, 2, 2, flatten=False)
        out = cnn(torch.rand(1, 1, 8, 8))
        assert (1, 2, 4, 4) == out.shape

    def test_shape_empty(self):
        cnn = CNN2d(1, 2, 3, 1, 2, 2)
        out = cnn(torch.rand(0, 1, 8, 8))
        assert (0, 32) == out.shape

    def test_shape_with_additional_dim(self):
        cnn = CNN2d(1, 2, 3, 1, 2, 2)
        out = cnn(torch.rand(1, 1, 1, 8, 8))
        assert (1, 1, 32) == out.shape
