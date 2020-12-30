import numpy as np
import torch

from mlprogram.nn import EmbeddingWithMask
from mlprogram.nn.treegen.embedding import (
    ActionSignatureEmbedding,
    ElementEmbedding,
    NlEmbedding,
)
from mlprogram.nn.utils.rnn import pad_sequence


class TestElementEmbedding(object):
    def test_parameters(self):
        e = ElementEmbedding(torch.nn.Embedding(1, 2), 3, 2, 5)
        pshape = {k: v.shape for k, v in e.named_parameters()}
        assert 2 == len(list(e.parameters()))
        assert (1, 2) == pshape["embed.weight"]
        assert (5, 2, 3) == pshape["elem_to_seq.weight"]

    def test_shape(self):
        e = ElementEmbedding(torch.nn.Embedding(1, 2), 3, 2, 5)
        input = torch.zeros(13, 1, 3, dtype=torch.long)
        output = e(input)
        assert (13, 1, 5) == output.shape

    def test_mask(self):
        e0 = ElementEmbedding(EmbeddingWithMask(4, 2, -1), 3, 2, 5)
        e1 = ElementEmbedding(EmbeddingWithMask(4, 2, -1), 4, 2, 5)
        e1.embed.weight.data = e0.embed.weight.data
        e1.elem_to_seq.weight.data[:, :, :3] = \
            e0.elem_to_seq.weight.data[:, :, :]

        input = torch.randint(3, [11, 1, 4])
        input[:, :, 3] = -1
        with torch.no_grad():
            output0 = e0(input[:, :, :3])
            output1 = e1(input)
        assert np.allclose(output0.numpy(), output1.numpy())


class TestActionSignatureEmbedding(object):
    def test_parameters(self):
        e = ActionSignatureEmbedding(2, 3, 11)
        pshape = {k: v.shape for k, v in e.named_parameters()}
        assert 2 == len(list(e.parameters()))
        assert (3, 11) == pshape["node_type_embed.weight"]
        assert (3, 11) == pshape["token_embed.weight"]

    def test_shape(self):
        e = ActionSignatureEmbedding(2, 3, 11)
        input = torch.zeros(13, 1, 6, 3, dtype=torch.long)
        output = e(input)
        assert (13, 1, 6, 11) == output.shape

    def test_token_mask(self):
        e = ActionSignatureEmbedding(2, 3, 11)
        input = torch.zeros(13, 1, 6, 3, dtype=torch.long)
        input[:, :, 1] = -1
        e(input)

    def test_node_type_mask(self):
        e = ActionSignatureEmbedding(2, 3, 11)
        input = torch.zeros(13, 1, 6, 3, dtype=torch.long)
        input[:, :, 0] = -1
        e(input)

    def test_reference_mask(self):
        e = ActionSignatureEmbedding(2, 3, 11)
        input = torch.zeros(13, 1, 6, 3, dtype=torch.long)
        input[:, :, 2] = -1
        e(input)

    def test_reference_embed(self):
        e = ActionSignatureEmbedding(2, 3, 11)
        input = torch.full(size=(2, 1, 6, 3), fill_value=-1, dtype=torch.long)
        input[0, :, :, 2] = 0
        input[1, :, :, 2] = 1
        with torch.no_grad():
            output = e(input)

        assert np.allclose(output[0].numpy(), output[1].numpy())


class TestNlEmbedding(object):
    def test_parameters(self):
        e = NlEmbedding(2, 3, 5, 7, 11)
        assert 3 == len(list(e.parameters()))

    def test_shape(self):
        e = NlEmbedding(2, 3, 5, 7, 11)
        w0 = torch.zeros(13).long()
        w1 = torch.zeros(11).long()
        w = pad_sequence([w0, w1], padding_value=-1)
        c0 = torch.zeros(13, 5).long()
        c1 = torch.zeros(11, 5).long()
        c = pad_sequence([c0, c1], padding_value=-1)
        e_w, e_c = e(w, c)
        assert (13, 2, 11) == e_w.data.shape
        assert (13, 2, 7) == e_c.data.shape
