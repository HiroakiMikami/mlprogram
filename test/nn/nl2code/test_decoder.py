import numpy as np
import torch

from mlprogram.nn.nl2code import Decoder, DecoderCell
from mlprogram.nn.nl2code.decoder import query_history
from mlprogram.nn.utils import rnn


class TestQueryHistory(object):
    def test_simple_case(self):
        # length = 3, minibatch-size = 2
        history = torch.FloatTensor([[[1], [-1]], [[2], [-2]], [[3], [-3]]])
        index = torch.LongTensor([0, 2])
        h = query_history(history, index)
        assert (2, 1) == h.shape
        assert np.array_equal([[1], [-3]], h.numpy())


class TestDecoderCell(object):
    def test_parameters(self):
        cell = DecoderCell(2, 3, 5, 7, 0.0)
        assert 8 == len(dict(cell.named_parameters()))

    def test_shape(self):
        cell = DecoderCell(2, 3, 5, 7, 0.0)
        ctx0 = torch.rand(3, 2)
        ctx1 = torch.rand(3, 1)
        ctx = rnn.pad_sequence([ctx0, ctx1])
        input = torch.rand(2, 3)
        parent_index = torch.randint(10, size=(2,))
        history = torch.rand(10, 2, 5)
        h_0 = torch.rand(2, 5)
        c_0 = torch.rand(2, 5)

        ctx, (h_1, c_1) = cell(ctx, input, parent_index, history, (h_0, c_0))
        assert (2, 2) == ctx.shape
        assert (2, 5) == h_1.shape
        assert (2, 5) == c_1.shape


class TestDecoder(object):
    def test_parameters(self):
        decoder = Decoder(2, 3, 5, 2, 3, 5, 7, 11, 0.0)
        assert 11 == len(dict(decoder.named_parameters()))

    def test_shape(self):
        decoder = Decoder(2, 3, 5, 2, 3, 2, 5, 7, 0.0)
        query0 = torch.rand(3, 2)
        query1 = torch.rand(1, 2)
        query = rnn.pad_sequence([query0, query1])
        action0 = torch.LongTensor([[0, 0, 0], [1, 1, 1], [1, 1, 1]])
        action1 = torch.LongTensor([[0, 0, 0]])
        actions = rnn.pad_sequence([action0, action1])  # (2, 2, 3)
        prev_action0 = torch.LongTensor([[0, 0, 0], [1, 1, 1], [1, 1, 1]])
        prev_action1 = torch.LongTensor([[0, 0, 0]])
        prev_actions = rnn.pad_sequence(
            [prev_action0, prev_action1])  # (2, 2, 3)
        history = torch.rand(10, 2, 5)
        h_0 = torch.rand(2, 5)
        c_0 = torch.rand(2, 5)

        output, contexts, history, h_n, c_n = decoder(
            nl_query_features=query,
            actions=actions,
            previous_actions=prev_actions,
            history=history,
            hidden_state=h_0,
            state=c_0,
        )
        assert (3, 2, 5) == output.data.shape
        assert np.array_equal(
            [[1, 1], [1, 0], [1, 0]], output.mask.numpy())
        assert (3, 2, 2) == contexts.data.shape
        assert np.array_equal(
            [[1, 1], [1, 0], [1, 0]], contexts.mask.numpy())
        assert (13, 2, 5) == history.shape
        assert (2, 5) == h_n.shape
        assert (2, 5) == c_n.shape
