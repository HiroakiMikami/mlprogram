import torch
import numpy as np
import unittest

from nl2prog.nn.treegen import ASTReaderBlock, ASTReader
from nl2prog.nn.utils.rnn import pad_sequence


class TestASTReaderBlock(unittest.TestCase):
    def test_parameters(self):
        block = ASTReaderBlock(2, 3, 1, 3, 0.0, 0)
        self.assertEqual(19, len(list(block.parameters())))

    def test_shape(self):
        block = ASTReaderBlock(2, 3, 1, 3, 0.0, 0)
        in0 = torch.Tensor(5, 3)
        in0 = pad_sequence([in0], 0)
        depth = torch.Tensor(5, 1)
        in1 = torch.Tensor(5, 1, 2)
        adj = torch.Tensor(1, 5, 5)
        out, weight = block(in0, depth, in1, adj)
        self.assertEqual((5, 1, 3), out.data.shape)
        self.assertEqual((5, 1), out.mask.shape)
        self.assertEqual((1, 5, 5), weight.shape)

    def test_mask(self):
        block = ASTReaderBlock(2, 3, 1, 3, 0.0, 0)
        in00 = torch.rand(5, 3)
        in01 = torch.rand(7, 3)
        depth = torch.randint(5, [7, 2])
        in1 = torch.rand(7, 2, 2)
        adj = torch.randint(1, [1, 7, 7]).bool().long()
        out0, weight0 = block(pad_sequence([in00, in01], 0), depth, in1, adj)
        out1, weight1 = block(pad_sequence([in00], 0), depth[:5, :1],
                              in1[:5, :1, :], adj[:1, :5, :5])
        out0 = out0.data[:5, :1, :]
        weight0 = weight0[:1, :5, :5]
        out1 = out1.data
        self.assertTrue(np.array_equal(out0.detach().numpy(),
                                       out1.detach().numpy()))
        self.assertTrue(np.array_equal(weight0.detach().numpy(),
                                       weight1.detach().numpy()))


class TestASTReader(unittest.TestCase):
    def test_parameters(self):
        reader = ASTReader(2, 3, 1, 3, 0.0, 5)
        self.assertEqual(19 * 5, len(list(reader.parameters())))

    def test_shape(self):
        reader = ASTReader(2, 3, 1, 3, 0.0, 5)
        in0 = torch.Tensor(5, 3)
        in0 = pad_sequence([in0], 0)
        depth = torch.Tensor(5, 1)
        in1 = torch.Tensor(5, 1, 2)
        adj = torch.Tensor(1, 5, 5)
        out = reader(in0, depth, in1, adj)
        self.assertEqual((5, 1, 3), out.data.shape)
        self.assertEqual((5, 1), out.mask.shape)

    def test_mask(self):
        block = ASTReader(2, 3, 1, 3, 0.0, 5)
        in00 = torch.rand(5, 3)
        in01 = torch.rand(7, 3)
        depth = torch.randint(5, [7, 2])
        in1 = torch.rand(7, 2, 2)
        adj = torch.randint(1, [1, 7, 7]).bool().long()
        out0 = block(pad_sequence([in00, in01], 0), depth, in1, adj)
        out1 = block(pad_sequence([in00], 0), depth[:5, :1],
                     in1[:5, :1, :], adj[:1, :5, :5])
        out0 = out0.data[:5, :1, :]
        out1 = out1.data
        self.assertTrue(np.array_equal(out0.detach().numpy(),
                                       out1.detach().numpy()))


if __name__ == "__main__":
    unittest.main()
