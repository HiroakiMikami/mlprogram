import unittest
import torch
import torch.nn as nn
import numpy as np

from nl2code.nn._embedding import EmbeddingWithMask, EmbeddingInverse


class TestEmbeddingWithMask(unittest.TestCase):
    def test_parameters(self):
        embedding = EmbeddingWithMask(2, 3, 0)
        params = dict(embedding.named_parameters())
        self.assertEqual(["weight"], list(params.keys()))
        self.assertEqual([(3, 3)], list(
            map(lambda x: x.shape, params.values())))

    def test_embedding(self):
        x = torch.LongTensor([1, 2])
        embedding = EmbeddingWithMask(2, 3, 0)
        torch.nn.init.eye_(embedding.weight)
        output = embedding(x)
        self.assertTrue(np.allclose(
            [[0, 1, 0], [0, 0, 1]], output.detach().numpy()))

    def test_mask(self):
        x = torch.LongTensor([1])
        embedding = EmbeddingWithMask(2, 3, 1)
        torch.nn.init.eye_(embedding.weight)
        output = embedding(x)
        self.assertTrue(np.allclose([[0, 0, 0]], output.detach().numpy()))


class TestEmbeddingInverse(unittest.TestCase):
    def test_parameters(self):
        e_without_bias = EmbeddingInverse(nn.Embedding(2, 3), False)
        params = dict(e_without_bias.named_parameters())
        self.assertEqual(0, len(params))

        e_with_bias = EmbeddingInverse(nn.Embedding(2, 3), True)
        params = dict(e_with_bias.named_parameters())
        self.assertEqual(1, len(params))
        self.assertEqual(["bias"], [key for key in params.keys()])
        self.assertEqual((2,), params["bias"].shape)

    def test_forward(self):
        e = nn.Embedding(3, 3)
        inv = EmbeddingInverse(e, False)
        x = torch.LongTensor([0, 1, 2])
        embed = e(x)
        x2 = inv(embed)
        self.assertEqual((3, 3), x2.shape)


if __name__ == "__main__":
    unittest.main()
