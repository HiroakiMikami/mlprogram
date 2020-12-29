import numpy as np
import torch

from mlprogram.nn import EmbeddingWithMask
from mlprogram.nn.treegen import (
    ActionEmbedding,
    ActionSignatureEmbedding,
    ElementEmbedding,
)


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


class TestActionEmbedding(object):
    def test_parameters(self):
        e = ActionEmbedding(1, 2, 3)
        pshape = {k: v.shape for k, v in e.named_parameters()}
        assert 2 == len(list(e.parameters()))
        assert (1, 3) == pshape["rule_embed.weight"]
        assert (3, 3) == pshape["token_embed.weight"]

    def test_shape(self):
        e = ActionEmbedding(1, 2, 3)
        out = e(torch.zeros(13, 1, 3, dtype=torch.long))
        assert (13, 1, 3) == out.shape

    def test_rule_mask(self):
        e = ActionEmbedding(1, 2, 3)
        input = torch.zeros(13, 1, 3, dtype=torch.long)
        input[:, :, 0] = -1
        e(input)

    def test_token_mask(self):
        e = ActionEmbedding(1, 2, 3)
        input = torch.zeros(13, 1, 3, dtype=torch.long)
        input[:, :, 1] = -1
        e(input)

    def test_reference_mask(self):
        e = ActionEmbedding(1, 2, 3)
        input = torch.zeros(13, 1, 3, dtype=torch.long)
        input[:, :, 2] = -1
        e(input)

    def test_reference_embed(self):
        e = ActionEmbedding(1, 2, 3)
        input = torch.zeros(13, 1, 3, dtype=torch.long)
        input[0, :, 2] = 0
        input[1, :, 2] = 1
        out = e(input)
        with torch.no_grad():
            out = e(input)

        assert np.allclose(out[0].numpy(), out[1].numpy())


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
