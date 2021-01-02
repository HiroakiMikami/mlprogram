import numpy as np
import torch

import mlprogram.nn.utils.rnn as rnn
from mlprogram.nn import BidirectionalLSTM


class TestBidirectionalLSTM(object):
    def test_parameters(self):
        encoder = BidirectionalLSTM(3, 14)
        assert 8 == len(list(encoder.named_parameters()))

    def test_shape(self):
        encoder = BidirectionalLSTM(3, 14)
        q0 = torch.rand(1, 3)
        q1 = torch.rand(3, 3)
        query = rnn.pad_sequence([q0, q1])
        output = encoder(query)
        assert (3, 2, 14) == output.data.shape

    def test_mask(self):
        encoder = BidirectionalLSTM(3, 14)
        q0 = torch.rand(2, 3)
        q1 = torch.rand(3, 3)
        query = rnn.pad_sequence([q0, q1])
        output = encoder(query)
        assert np.all(output.data[2, 0, :].detach().numpy() == 0)
