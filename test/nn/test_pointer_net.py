import torch
import numpy as np

from mlprogram.nn import PointerNet
from mlprogram.nn.utils.rnn import pad_sequence


class TestPointerNet(object):
    def test_parameters(self):
        layer = PointerNet(1, 2, 3)
        params = dict(layer.named_parameters())
        assert 6 == len(params)
        assert (3, 1) == params["w1.weight"].shape
        assert (3, 2) == params["w2.weight"].shape
        assert (1, 3) == params["v.weight"].shape

    def test_pointer_net(self):
        value0 = torch.FloatTensor([[1]])
        value1 = torch.FloatTensor([[1], [1], [1], [1]])
        value = pad_sequence([value0, value1])

        key0 = torch.FloatTensor([[0]])
        key1 = torch.FloatTensor([[0]])
        key = pad_sequence([key0, key1]).data

        layer = PointerNet(1, 1, 2)
        log_output = layer(key, value)  # (1, 2, 4)
        output = torch.exp(log_output)
        output *= value.mask.permute(1, 0).view(1, 2, 4).float()
        assert (1, 2, 4) == output.shape
        assert np.allclose(
            [[1, 0, 0, 0], [0.25, 0.25, 0.25, 0.25]],
            output.detach().numpy())
