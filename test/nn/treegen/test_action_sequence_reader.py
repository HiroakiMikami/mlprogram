import torch
import numpy as np
import unittest

from mlprogram.nn.treegen \
    import ActionSequenceReaderBlock, ActionSequenceReader
from mlprogram.nn.utils.rnn import pad_sequence


class TestActionSequenceReaderBlock(unittest.TestCase):
    def test_parameters(self):
        block = ActionSequenceReaderBlock(2, 3, 1, 3, 0.0, 0)
        self.assertEqual(19, len(list(block.parameters())))

    def test_shape(self):
        block = ActionSequenceReaderBlock(2, 3, 1, 3, 0.0, 0)
        in0 = torch.Tensor(5, 3)
        in0 = pad_sequence([in0], 0)
        depth = torch.Tensor(5, 1)
        in1 = torch.Tensor(5, 1, 2)
        adj = torch.Tensor(1, 5, 5)
        out, weight = block(in0, depth, in1, adj)
        self.assertEqual((5, 1, 3), out.data.shape)
        self.assertEqual((5, 1), out.mask.shape)
        self.assertEqual((1, 5, 5), weight.shape)

    def test_dependency(self):
        torch.manual_seed(0)
        block = ActionSequenceReaderBlock(2, 3, 1, 3, 0.0, 0)
        in0 = torch.rand(3, 3)
        depth = torch.randint(3, [3, 1])
        in1 = torch.rand(3, 1, 2)
        adj = torch.randint(1, [1, 3, 3]).bool().long()
        out0, weight0 = block(pad_sequence([in0], 0), depth, in1, adj)
        out1, weight1 = block(pad_sequence([in0[:2, :]], 0), depth[:2, :],
                              in1[:2, :, :], adj[:, :2, :2])
        out0 = out0.data[:2, :, :]
        weight0 = weight0[:1, :2, :2]
        out1 = out1.data
        self.assertTrue(np.allclose(out0.detach().numpy(),
                                    out1.detach().numpy()))
        self.assertTrue(np.allclose(weight0.detach().numpy(),
                                    weight1.detach().numpy()))

    def test_mask(self):
        torch.manual_seed(0)
        block = ActionSequenceReaderBlock(2, 3, 1, 3, 0.0, 0)
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
        self.assertTrue(np.allclose(out0.detach().numpy(),
                                    out1.detach().numpy()))
        self.assertTrue(np.allclose(weight0.detach().numpy(),
                                    weight1.detach().numpy()))


class TestActionSequenceReader(unittest.TestCase):
    def test_parameters(self):
        reader = ActionSequenceReader(1, 1, 1, 3, 2, 3, 1, 3, 0.0, 5)
        self.assertEqual(19 * 5 + 5, len(list(reader.parameters())))

    def test_shape(self):
        reader = ActionSequenceReader(1, 1, 1, 3, 2, 3, 1, 3, 0.0, 5)
        in0 = torch.zeros(5, 3).long()
        in0 = pad_sequence([in0], 0)
        depth = torch.Tensor(5, 1)
        in1 = torch.zeros(5, 4, 3).long()
        in1 = pad_sequence([in1], 0)
        adj = torch.Tensor(1, 5, 5)
        out = reader({
            "previous_actions": in0,
            "previous_action_rules": in1,
            "depthes": depth,
            "adjacency_matrix": adj
        })["action_features"]
        self.assertEqual((5, 1, 3), out.data.shape)
        self.assertEqual((5, 1), out.mask.shape)

    def test_mask(self):
        torch.manual_seed(0)
        reader = ActionSequenceReader(1, 1, 1, 3, 2, 3, 1, 3, 0.0, 5)
        in00 = torch.zeros(5, 3).long()
        in01 = torch.zeros(7, 3).long()
        depth = torch.randint(5, [7, 2])
        in10 = torch.zeros(5, 4, 3).long()
        in11 = torch.zeros(7, 4, 3).long()
        adj = torch.randint(1, [2, 7, 7]).bool().long()
        out0 = reader({
            "previous_actions": pad_sequence([in00, in01], 0),
            "previous_action_rules": pad_sequence([in10, in11], 0),
            "depthes": depth,
            "adjacency_matrix": adj
        })["action_features"]
        out1 = reader({
            "previous_actions": pad_sequence([in00], 0),
            "previous_action_rules": pad_sequence([in10], 0),
            "depthes": depth[:5, :1],
            "adjacency_matrix": adj[:1, :5, :5]
        })["action_features"]
        out0 = out0.data[:5, :1, :]
        out1 = out1.data
        self.assertTrue(np.allclose(out0.detach().numpy(),
                                    out1.detach().numpy()))


if __name__ == "__main__":
    unittest.main()
