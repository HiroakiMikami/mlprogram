import numpy as np
import torch

from mlprogram.nn.treegen import ActionSequenceReaderBlock, Decoder, DecoderBlock
from mlprogram.nn.utils.rnn import pad_sequence


class TestDecoderBlock(object):
    def test_parameters(self):
        block = DecoderBlock(1, 3, 5, 1, 0.0)
        assert 18 == len(list(block.parameters()))

    def test_shape(self):
        block = DecoderBlock(1, 3, 5, 1, 0.0)
        query0 = torch.Tensor(7, 1)
        nl0 = torch.Tensor(11, 1)
        ast0 = torch.Tensor(7, 1)
        out, w0, w1 = block(pad_sequence([query0], 0),
                            pad_sequence([nl0], 0),
                            pad_sequence([ast0], 0))
        assert (7, 1, 5) == out.data.shape
        assert (7, 1) == out.mask.shape
        assert (1, 7, 11) == w0.shape
        assert (1, 7, 7) == w1.shape

    def test_mask_nl(self):
        block = DecoderBlock(1, 3, 5, 1, 0.0)
        query0 = torch.rand(7, 1)
        nl0 = torch.rand(11, 1)
        nl1 = torch.rand(13, 1)
        ast0 = torch.rand(7, 1)
        out0, w00, w01 = block(pad_sequence([query0], 0),
                               pad_sequence([nl0], 0),
                               pad_sequence([ast0], 0))
        out1, w10, w11 = block(pad_sequence([query0, query0], 0),
                               pad_sequence([nl0, nl1], 0),
                               pad_sequence([ast0, ast0], 0))
        out0 = out0.data
        out1 = out1.data[:7, :1, :]
        w10 = w10[:1, :7, :11]
        w11 = w11[:1, :7, :7]
        assert np.allclose(out0.detach().numpy(),
                           out1.detach().numpy())
        assert np.allclose(w00.detach().numpy(),
                           w10.detach().numpy())
        assert np.allclose(w01.detach().numpy(),
                           w11.detach().numpy())

    def test_mask_ast_and_query(self):
        block = DecoderBlock(1, 3, 5, 1, 0.0)
        query0 = torch.rand(7, 1)
        query1 = torch.rand(9, 1)
        nl0 = torch.rand(11, 1)
        ast0 = torch.rand(7, 1)
        ast1 = torch.rand(9, 1)
        out0, w00, w01 = block(pad_sequence([query0], 0),
                               pad_sequence([nl0], 0),
                               pad_sequence([ast0], 0))
        out1, w10, w11 = block(pad_sequence([query0, query1], 0),
                               pad_sequence([nl0, nl0], 0),
                               pad_sequence([ast0, ast1], 0))
        out0 = out0.data
        out1 = out1.data[:7, :1, :]
        w10 = w10[:1, :7, :11]
        w11 = w11[:1, :7, :7]
        assert np.allclose(out0.detach().numpy(),
                           out1.detach().numpy())
        assert np.allclose(w00.detach().numpy(),
                           w10.detach().numpy())
        assert np.allclose(w01.detach().numpy(),
                           w11.detach().numpy())

    def test_attn_mask(self):
        block = DecoderBlock(1, 3, 5, 1, 0.0)
        query0 = torch.rand(7, 1)
        nl0 = torch.rand(11, 1)
        ast0 = torch.rand(7, 1)
        out0, w00, w01 = block(pad_sequence([query0[:5, :]], 0),
                               pad_sequence([nl0], 0),
                               pad_sequence([ast0[:5, :]], 0))
        out1, w10, w11 = block(pad_sequence([query0], 0),
                               pad_sequence([nl0], 0),
                               pad_sequence([ast0], 0))
        out0 = out0.data
        out1 = out1.data[:5, :1, :]
        w10 = w10[:1, :5, :11]
        w11 = w11[:1, :5, :5]
        assert np.allclose(out0.detach().numpy(),
                           out1.detach().numpy())
        assert np.allclose(w00.detach().numpy(),
                           w10.detach().numpy())
        assert np.allclose(w01.detach().numpy(),
                           w11.detach().numpy())


class TestActionSequenceReaderBlock(object):
    def test_parameters(self):
        block = ActionSequenceReaderBlock(2, 3, 1, 3, 0.0, 0)
        assert 19 == len(list(block.parameters()))

    def test_shape(self):
        block = ActionSequenceReaderBlock(2, 3, 1, 3, 0.0, 0)
        in0 = torch.Tensor(5, 3)
        in0 = pad_sequence([in0], 0)
        depth = torch.Tensor(5, 1)
        in1 = torch.Tensor(5, 1, 2)
        adj = torch.Tensor(1, 5, 5)
        out, weight = block(in0, depth, in1, adj)
        assert (5, 1, 3) == out.data.shape
        assert (5, 1) == out.mask.shape
        assert (1, 5, 5) == weight.shape

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
        assert np.allclose(out0.detach().numpy(),
                           out1.detach().numpy())
        assert np.allclose(weight0.detach().numpy(),
                           weight1.detach().numpy())

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
        assert np.allclose(out0.detach().numpy(),
                           out1.detach().numpy())
        assert np.allclose(weight0.detach().numpy(),
                           weight1.detach().numpy())


class TestDecoder(object):
    def test_parameters(self):
        decoder = Decoder(1, 3, 5, 1, 3, 1, 0.0, 5, 5)
        assert 185 == len(list(decoder.parameters()))

    def test_shape(self):
        decoder = Decoder(1, 1, 5, 3, 3, 1, 0.0, 5, 5)
        in0 = torch.rand(5, 1)
        in0 = pad_sequence([in0], 0)
        depth = torch.Tensor(5, 1)
        in1 = torch.rand(5, 1)
        in1 = pad_sequence([in1], 0)
        adj = torch.Tensor(1, 5, 5)
        query0 = torch.rand(5, 1)
        nl0 = torch.Tensor(11, 1)
        out = decoder(
            action_query_features=pad_sequence([query0], 0),
            nl_query_features=pad_sequence([nl0], 0),
            action_features=in0,
            action_rule_features=in1,
            depthes=depth,
            adjacency_matrix=adj,
        )
        assert (5, 1, 3) == out.data.shape
        assert (5, 1) == out.mask.shape

    def test_mask(self):
        torch.manual_seed(0)
        decoder = Decoder(1, 1, 5, 3, 3, 1, 0.0, 5, 5)
        in00 = torch.rand(5, 1)
        in01 = torch.rand(7, 1)
        depth = torch.randint(1, [7, 2])
        in10 = torch.rand(5, 1)
        in11 = torch.rand(7, 1)
        adj = torch.randint(1, [2, 7, 7]).bool().float()
        query00 = torch.zeros(5, 1)
        query01 = torch.zeros(7, 1)
        nl0 = torch.rand(11, 1)
        out0 = decoder(
            nl_query_features=pad_sequence([nl0, nl0], 0),
            action_query_features=pad_sequence([query00, query01], 0),
            action_features=pad_sequence([in00, in01], 0),
            action_rule_features=pad_sequence([in10, in11], 0),
            depthes=depth,
            adjacency_matrix=adj
        )
        out1 = decoder(
            nl_query_features=pad_sequence([nl0], 0),
            action_query_features=pad_sequence([query00], 0),
            action_features=pad_sequence([in00], 0),
            action_rule_features=pad_sequence([in10], 0),
            depthes=depth[:5, :1],
            adjacency_matrix=adj[:1, :5, :5]
        )
        out0 = out0.data[:5, :1, :]
        out1 = out1.data
        assert np.allclose(out0.detach().numpy(),
                           out1.detach().numpy())
