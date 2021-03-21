import numpy as np
import pytest
import torch

from mlprogram.nn.action_sequence.lstm_decoder import CatInput
from mlprogram.nn.action_sequence.lstm_tree_decoder import (
    LSTMTreeDecoder,
    query_history,
)
from mlprogram.nn.utils import rnn


@pytest.fixture
def decoder():
    return LSTMTreeDecoder(
        inject_input=CatInput(),
        input_feature_size=2,
        action_feature_size=3,
        output_feature_size=5,
        dropout=0.0
    )


class TestQueryHistory(object):
    def test_simple_case(self):
        # length = 3, minibatch-size = 2
        history = torch.FloatTensor([[[1], [-1]], [[2], [-2]], [[3], [-3]]])
        index = torch.LongTensor([0, 2])
        h = query_history(history, index)
        assert (2, 1) == h.shape
        assert np.array_equal([[1], [-3]], h.numpy())


class TestLSTMTreeDecoder(object):
    def test_parameters(self, decoder):
        assert 4 == len(dict(decoder.named_parameters()))

    def test_shape(self, decoder):
        query = torch.rand(2, 2)
        action0 = torch.LongTensor([[0, 0, 0], [1, 1, 1], [1, 1, 1]])
        action1 = torch.LongTensor([[0, 0, 0]])
        actions = rnn.pad_sequence([action0, action1])  # (2, 2, 3)
        prev_action0 = torch.rand(3, 3)
        prev_action1 = torch.rand(1, 3)
        prev_actions = rnn.pad_sequence([prev_action0, prev_action1])
        history = torch.rand(10, 2, 5)
        h_0 = torch.rand(2, 5)
        c_0 = torch.rand(2, 5)

        output, history, h_n, c_n = decoder(
            input_feature=query,
            actions=actions,
            action_features=prev_actions,
            history=history,
            hidden_state=h_0,
            state=c_0,
        )
        assert (3, 2, 5) == output.data.shape
        assert np.array_equal(
            [[1, 1], [1, 0], [1, 0]], output.mask.numpy())
        assert (13, 2, 5) == history.shape
        assert (2, 5) == h_n.shape
        assert (2, 5) == c_n.shape
