import numpy as np
import pytest
import torch

from mlprogram.nn.action_sequence import AttentionInput, CatInput, LSTMDecoder
from mlprogram.nn.utils import rnn


@pytest.fixture
def decoder():
    return LSTMDecoder(
        inject_input=CatInput(),
        input_feature_size=2,
        action_feature_size=3,
        output_feature_size=5,
        dropout=0.0)


@pytest.fixture
def decoder_with_attention_input():
    return LSTMDecoder(
        inject_input=AttentionInput(attn_hidden_size=7),
        input_feature_size=2,
        action_feature_size=3,
        output_feature_size=5,
        dropout=0.0)


class TestLSTMDecoder(object):
    def test_parameters(self, decoder):
        assert 4 == len(dict(decoder.named_parameters()))

    def test_shape(self, decoder):
        input = torch.rand(2, 2)
        action0 = torch.rand(3, 3)
        action1 = torch.rand(1, 3)
        action = rnn.pad_sequence([action0, action1])
        h_0 = torch.rand(2, 5)
        c_0 = torch.rand(2, 5)

        output, h_n, c_n = decoder(
            input_feature=input,
            action_features=action,
            hidden_state=h_0,
            state=c_0
        )
        assert (3, 2, 5) == output.data.shape
        assert np.array_equal([[1, 1], [1, 0], [1, 0]], output.mask.numpy())
        assert (2, 5) == h_n.shape
        assert (2, 5) == c_n.shape

    def test_state(self, decoder):
        input = torch.rand(1, 2)
        action0 = torch.ones(2, 3)
        action = rnn.pad_sequence([action0])
        h_0 = torch.zeros(1, 5)
        c_0 = torch.zeros(1, 5)

        output, _, _ = decoder(
            input_feature=input,
            action_features=action,
            hidden_state=h_0,
            state=c_0
        )
        assert not np.array_equal(
            output.data[0, 0, :].detach().numpy(), output.data[1, 0, :].detach().numpy()
        )

    def test_attention_input(self, decoder_with_attention_input):
        input0 = torch.rand(3, 2)
        input1 = torch.rand(1, 2)
        input = rnn.pad_sequence([input0, input1])
        action0 = torch.rand(3, 3)
        action1 = torch.rand(1, 3)
        action = rnn.pad_sequence([action0, action1])
        h_0 = torch.rand(2, 5)
        c_0 = torch.rand(2, 5)

        output, h_n, c_n = decoder_with_attention_input(
            input_feature=input,
            action_features=action,
            hidden_state=h_0,
            state=c_0
        )
        assert (3, 2, 5) == output.data.shape
        assert np.array_equal([[1, 1], [1, 0], [1, 0]], output.mask.numpy())
        assert (2, 5) == h_n.shape
        assert (2, 5) == c_n.shape

        output2, h_n2, c_n2 = decoder_with_attention_input(
            input_feature=rnn.pad_sequence([input1]),
            action_features=rnn.pad_sequence([action1]),
            hidden_state=h_0[1:, :],
            state=c_0[1:, :]
        )

        assert np.allclose(output.data[:1, 1, :].detach().numpy(),
                           output2.data[:, 0, :].detach().numpy())
