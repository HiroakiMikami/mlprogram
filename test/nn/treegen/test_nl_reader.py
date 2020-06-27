import torch
import numpy as np
import unittest

from mlprogram.nn.treegen import NLReaderBlock, NLReader
from mlprogram.nn.utils.rnn import pad_sequence


class TestNLReaderBlock(unittest.TestCase):
    def test_parameters(self):
        block = NLReaderBlock(2, 3, 1, 0.0, 0)
        self.assertEqual(21, len(list(block.parameters())))

    def test_shape(self):
        block = NLReaderBlock(2, 3, 1, 0.0, 0)
        in0 = torch.Tensor(5, 3)
        in0 = pad_sequence([in0], 0)
        in1 = torch.Tensor(5, 1, 2)
        out, weight = block(in0, in1)
        self.assertEqual((5, 1, 3), out.data.shape)
        self.assertEqual((5, 1), out.mask.shape)
        self.assertEqual((1, 5, 5), weight.shape)

    def test_mask(self):
        block = NLReaderBlock(2, 3, 1, 0.0, 0)
        in00 = torch.rand(5, 3)
        in01 = torch.rand(7, 3)
        in1 = torch.rand(7, 2, 2)
        out0, weight0 = block(pad_sequence([in00, in01], 0), in1)
        out1, weight1 = block(pad_sequence([in00], 0), in1[:5, :1, :])
        out0 = out0.data[:5, :1, :]
        weight0 = weight0[:1, :5, :5]
        out1 = out1.data
        self.assertTrue(np.allclose(out0.detach().numpy(),
                                    out1.detach().numpy()))
        self.assertTrue(np.allclose(weight0.detach().numpy(),
                                    weight1.detach().numpy()))


class TestNLReader(unittest.TestCase):
    def test_parameters(self):
        reader = NLReader(1, 1, 7, 2, 3, 1, 0.0, 5)
        self.assertEqual(3 + 21 * 5, len(list(reader.parameters())))

    def test_shape(self):
        reader = NLReader(1, 1, 7, 2, 3, 1, 0.0, 5)
        in0 = torch.zeros(5).long()
        in0 = pad_sequence([in0], 0)
        in1 = torch.zeros(5, 7).long()
        in1 = pad_sequence([in1], 0)
        out = reader({"word_nl_query": in0,
                      "char_nl_query": in1})["nl_query_features"]
        self.assertEqual((5, 1, 3), out.data.shape)
        self.assertEqual((5, 1), out.mask.shape)

    def test_mask(self):
        reader = NLReader(1, 1, 7, 2, 3, 1, 0.0, 5)
        in00 = torch.zeros(5).long()
        in01 = torch.zeros(7).long()
        in10 = torch.zeros(5, 7).long()
        in11 = torch.zeros(7, 7).long()
        out0 = reader({"word_nl_query": pad_sequence([in00, in01], 0),
                       "char_nl_query": pad_sequence([in10, in11])}
                      )["nl_query_features"]
        out1 = reader({"word_nl_query": pad_sequence([in00], 0),
                       "char_nl_query": pad_sequence([in10])
                       })["nl_query_features"]
        out0 = out0.data[:5, :1, :]
        out1 = out1.data
        self.assertTrue(np.allclose(out0.detach().numpy(),
                                    out1.detach().numpy()))


if __name__ == "__main__":
    unittest.main()
