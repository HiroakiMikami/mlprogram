import numpy as np
import torch

from mlprogram.nn.action_sequence import ActionsEmbedding, PreviousActionsEmbedding
from mlprogram.nn.utils.rnn import pad_sequence


class TestPreviousActionsEmbedding(object):
    def test_parameters(self):
        e = PreviousActionsEmbedding(1, 2, 3)
        pshape = {k: v.shape for k, v in e.named_parameters()}
        assert 2 == len(list(e.parameters()))
        assert (1, 3) == pshape["rule_embed.weight"]
        assert (3, 3) == pshape["token_embed.weight"]

    def test_shape(self):
        e = PreviousActionsEmbedding(1, 2, 3)
        out = e(pad_sequence([torch.zeros(13, 3, dtype=torch.long)]))
        assert (13, 1, 3) == out.data.shape

    def test_rule_mask(self):
        e = PreviousActionsEmbedding(1, 2, 3)
        input = torch.zeros(13, 3, dtype=torch.long)
        input[:, 0] = -1
        e(pad_sequence([input]))

    def test_token_mask(self):
        e = PreviousActionsEmbedding(1, 2, 3)
        input = torch.zeros(13, 3, dtype=torch.long)
        input[:, 1] = -1
        e(pad_sequence([input]))

    def test_reference_mask(self):
        e = PreviousActionsEmbedding(1, 2, 3)
        input = torch.zeros(13, 3, dtype=torch.long)
        input[:, 2] = -1
        e(pad_sequence([input]))

    def test_reference_embed(self):
        e = PreviousActionsEmbedding(1, 2, 3)
        input = torch.zeros(13, 3, dtype=torch.long)
        input[0, 2] = 0
        input[1, 2] = 1
        with torch.no_grad():
            out = e(pad_sequence([input])).data

        assert np.allclose(out[0].numpy(), out[1].numpy())


class TestActionsEmbedding(object):
    def test_parameters(self):
        e = ActionsEmbedding(1, 2, 3, 4, 5)
        assert 3 == len(list(e.parameters()))

    def test_shape(self):
        e = ActionsEmbedding(1, 2, 3, 4, 5)
        out = e(
            pad_sequence([torch.zeros(13, 3, dtype=torch.long)]),
            pad_sequence([torch.zeros(13, 3, dtype=torch.long)]),
        )
        assert (13, 1, 14) == out.data.shape
