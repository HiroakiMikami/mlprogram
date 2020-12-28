import numpy as np
import torch

from mlprogram.nn.action_sequence import LSTMDecoder
from mlprogram.nn.utils import rnn


class TestLSTMDecoder(object):
    def test_parameters(self):
        decoder = LSTMDecoder(2, 3, 2, 3, 5, 0.0)
        assert 6 == len(dict(decoder.named_parameters()))

    def test_shape(self):
        decoder = LSTMDecoder(2, 3, 2, 3, 5, 0.0)
        input = torch.rand(2, 2)
        prev_action0 = torch.LongTensor([[0, 0, 0], [1, 1, 1], [1, 1, 1]])
        prev_action1 = torch.LongTensor([[0, 0, 0]])
        prev_action = rnn.pad_sequence(
            [prev_action0, prev_action1])  # (2, 2, 3)
        h_0 = torch.rand(2, 5)
        c_0 = torch.rand(2, 5)

        output, h_n, c_n = decoder(
            input_feature=input,
            previous_actions=prev_action,
            hidden_state=h_0,
            state=c_0
        )
        assert (3, 2, 5) == output.data.shape
        assert np.array_equal([[1, 1], [1, 0], [1, 0]], output.mask.numpy())
        assert (2, 5) == h_n.shape
        assert (2, 5) == c_n.shape

    def test_state(self):
        decoder = LSTMDecoder(2, 3, 2, 3, 5, 0.0)
        input = torch.rand(1, 2)
        action0 = torch.LongTensor([[0, 0, 0], [1, 1, 1]])
        action = rnn.pad_sequence([action0])
        h_0 = torch.zeros(1, 5)
        c_0 = torch.zeros(1, 5)

        output, _, _ = decoder(
            input_feature=input,
            previous_actions=action,
            hidden_state=h_0,
            state=c_0
        )
        assert not np.array_equal(
            output.data[0, 0, :].detach().numpy(), output.data[1, 0, :].detach().numpy()
        )
