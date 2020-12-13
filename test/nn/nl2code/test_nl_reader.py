import numpy as np
import torch

import mlprogram.nn.utils.rnn as rnn
from mlprogram.builtins import Environment
from mlprogram.nn.nl2code import NLReader


class TestNLReader(object):
    def test_parameters(self):
        encoder = NLReader(5, 3, 14)
        assert 9 == len(list(encoder.named_parameters()))

    def test_shape(self):
        encoder = NLReader(5, 3, 14)
        q0 = torch.LongTensor([1, 2])
        q1 = torch.LongTensor([1, 2, 3])
        query = rnn.pad_sequence([q0, q1])
        output = encoder(Environment({"word_nl_query": query}))["nl_query_features"]
        assert (3, 2, 14) == output.data.shape

    def test_mask(self):
        encoder = NLReader(5, 3, 14)
        q0 = torch.LongTensor([1, 2])
        q1 = torch.LongTensor([1, 2, 3])
        query = rnn.pad_sequence([q0, q1])
        output = encoder(Environment({"word_nl_query": query}))["nl_query_features"]
        assert np.all(output.data[2, 0, :].detach().numpy() == 0)
