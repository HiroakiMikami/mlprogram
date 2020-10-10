import torch
import numpy as np

from mlprogram.nn.action_sequence import RnnDecoder
from mlprogram.nn.utils import rnn


class TestRnnDecoder(object):
    def test_parameters(self):
        decoder = RnnDecoder(2, 3, 5, 0.0)
        assert 4 == len(dict(decoder.named_parameters()))

    def test_shape(self):
        decoder = RnnDecoder(2, 3, 5, 0.0)
        input = torch.rand(2, 2)
        action0 = torch.rand(3, 3)  # length = 3
        action1 = torch.rand(1, 3)  # length = 1
        action = rnn.pad_sequence([action0, action1])
        h_0 = torch.rand(2, 5)
        c_0 = torch.rand(2, 5)

        inputs = decoder({
            "input_feature": input,
            "action_features": action,
            "hidden_state": h_0,
            "state": c_0
        })
        output = inputs["action_features"]
        h_n = inputs["hidden_state"]
        c_n = inputs["state"]
        assert (3, 2, 5) == output.data.shape
        assert np.array_equal(
            [[1, 1], [1, 0], [1, 0]], output.mask.numpy())
        assert (2, 5) == h_n.shape
        assert (2, 5) == c_n.shape

    def test_state(self):
        decoder = RnnDecoder(2, 3, 5, 0.0)
        input = torch.rand(1, 2)
        action0 = torch.ones(2, 3)  # length = 3
        action = rnn.pad_sequence([action0])
        h_0 = torch.zeros(1, 5)
        c_0 = torch.zeros(1, 5)

        inputs = decoder({
            "input_feature": input,
            "action_features": action,
            "hidden_state": h_0,
            "state": c_0
        })
        output = inputs["action_features"].data
        assert not np.array_equal(
            output[0, 0, :].detach().numpy(), output[1, 0, :].detach().numpy()
        )
