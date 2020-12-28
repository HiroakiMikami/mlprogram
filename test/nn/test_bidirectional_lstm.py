import numpy as np
import torch

import mlprogram.nn.utils.rnn as rnn
from mlprogram.nn import BidirectionalLSTM


class TestBidirectionalLSTM(object):
    def test_parameters(self):
        encoder = BidirectionalLSTM(5, 3, 14)
        assert 9 == len(list(encoder.named_parameters()))

    def test_shape(self):
        encoder = BidirectionalLSTM(5, 3, 14)
        q0 = torch.LongTensor([1, 2])
        q1 = torch.LongTensor([1, 2, 3])
        query = rnn.pad_sequence([q0, q1])
        output = encoder(query)
        assert (3, 2, 14) == output.data.shape

    def test_mask(self):
        encoder = BidirectionalLSTM(5, 3, 14)
        q0 = torch.LongTensor([1, 2])
        q1 = torch.LongTensor([1, 2, 3])
        query = rnn.pad_sequence([q0, q1])
        output = encoder(query)
        assert np.all(output.data[2, 0, :].detach().numpy() == 0)
