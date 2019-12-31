import torch
import numpy as np
import unittest

from nl2prog.nn.treegen import NLReaderBlock, NLReader
from nl2prog.nn.utils.rnn import pad_sequence


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
        self.assertTrue(np.array_equal(out0.detach().numpy(),
                                       out1.detach().numpy()))
        self.assertTrue(np.array_equal(weight0.detach().numpy(),
                                       weight1.detach().numpy()))


class TestNLReader(unittest.TestCase):
    def test_parameters(self):
        reader = NLReader(2, 3, 1, 0.0, 5)
        self.assertEqual(21 * 5, len(list(reader.parameters())))

    def test_shape(self):
        reader = NLReader(2, 3, 1, 0.0, 5)
        in0 = torch.Tensor(5, 3)
        in0 = pad_sequence([in0], 0)
        in1 = torch.Tensor(5, 1, 2)
        out = reader(in0, in1)
        self.assertEqual((5, 1, 3), out.data.shape)
        self.assertEqual((5, 1), out.mask.shape)

    def test_mask(self):
        reader = NLReader(2, 3, 1, 0.0, 5)
        in00 = torch.rand(5, 3)
        in01 = torch.rand(7, 3)
        in1 = torch.rand(7, 2, 2)
        out0 = reader(pad_sequence([in00, in01], 0), in1)
        out1 = reader(pad_sequence([in00], 0), in1[:5, :1, :])
        out0 = out0.data[:5, :1, :]
        out1 = out1.data
        self.assertTrue(np.array_equal(out0.detach().numpy(),
                                       out1.detach().numpy()))


if __name__ == "__main__":
    unittest.main()
