import numpy as np
import torch
import torch.nn as nn

from mlprogram.nn.embedding import EmbeddingInverse, EmbeddingWithMask
from mlprogram.nn.utils.rnn import pad_sequence


class TestEmbeddingWithMask(object):
    def test_parameters(self):
        embedding = EmbeddingWithMask(2, 3, -1)
        params = dict(embedding.named_parameters())
        assert ["weight"] == list(params.keys())
        assert [(2, 3)] == list(
            map(lambda x: x.shape, params.values()))

    def test_embedding_sequence(self):
        x0 = torch.LongTensor([0, 1])
        x1 = torch.LongTensor([0, 1, 1])
        x = pad_sequence([x0, x1], padding_value=-1)
        embedding = EmbeddingWithMask(2, 2, -1)
        torch.nn.init.eye_(embedding.weight)
        output = embedding(x)
        assert np.allclose([[1, 0], [0, 1], [0, 0]],
                           output.data[:, 0, :].detach().numpy())
        assert np.array_equal([[1, 1], [1, 1], [0, 1]],
                              output.mask.detach().numpy())

    def test_embedding(self):
        x = torch.LongTensor([0, 1, -1])
        embedding = EmbeddingWithMask(2, 2, -1)
        torch.nn.init.eye_(embedding.weight)
        output = embedding(x)
        assert np.allclose(
            [[1, 0], [0, 1], [0, 0]], output.detach().numpy())

    def test_mask(self):
        x = torch.LongTensor([-1])
        embedding = EmbeddingWithMask(2, 3, -1)
        torch.nn.init.eye_(embedding.weight)
        output = embedding(x)
        assert np.allclose([[0, 0, 0]], output.detach().numpy())


class TestEmbeddingInverse(object):
    def test_parameters(self):
        e_without_bias = EmbeddingInverse(2, False)
        params = dict(e_without_bias.named_parameters())
        assert 0 == len(params)

        e_with_bias = EmbeddingInverse(2, True)
        params = dict(e_with_bias.named_parameters())
        assert 1 == len(params)
        assert ["bias"] == [key for key in params.keys()]
        assert (2,) == params["bias"].shape

    def test_forward(self):
        e = nn.Embedding(3, 3)
        inv = EmbeddingInverse(3, False)
        x = torch.LongTensor([0, 1, 2])
        embed = e(x)
        x2 = inv(embed, e)
        assert (3, 3) == x2.shape
