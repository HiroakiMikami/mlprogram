import numpy as np
import torch

from mlprogram.nn.treegen import Encoder, EncoderBlock
from mlprogram.nn.utils.rnn import pad_sequence


class TestEncoderBlock(object):
    def setup(self):
        torch.manual_seed(0)

    def test_parameters(self):
        block = EncoderBlock(2, 3, 1, 0.0, 0)
        assert 21 == len(list(block.parameters()))

    def test_shape(self):
        block = EncoderBlock(2, 3, 1, 0.0, 0)
        in0 = torch.Tensor(5, 3)
        in0 = pad_sequence([in0], 0)
        in1 = torch.Tensor(5, 1, 2)
        out, weight = block(in0, in1)
        assert (5, 1, 3) == out.data.shape
        assert (5, 1) == out.mask.shape
        assert (1, 5, 5) == weight.shape

    def test_mask(self):
        block = EncoderBlock(2, 3, 1, 0.0, 0)
        in00 = torch.rand(5, 3)
        in01 = torch.rand(7, 3)
        in1 = torch.rand(7, 2, 2)
        out0, weight0 = block(pad_sequence([in00, in01], 0), in1)
        out1, weight1 = block(pad_sequence([in00], 0), in1[:5, :1, :])
        out0 = out0.data[:5, :1, :]
        weight0 = weight0[:1, :5, :5]
        out1 = out1.data
        assert np.allclose(out0.detach().numpy(),
                           out1.detach().numpy())
        assert np.allclose(weight0.detach().numpy(),
                           weight1.detach().numpy())


class TestEncoder(object):
    def test_parameters(self):
        reader = Encoder(2, 3, 1, 0.0, 5)
        assert 105 == len(list(reader.parameters()))

    def test_shape(self):
        reader = Encoder(2, 3, 1, 0.0, 5)
        in0 = torch.rand(5, 3)
        in0 = pad_sequence([in0], 0)
        in1 = torch.rand(5, 2)
        in1 = pad_sequence([in1], 0)
        out = reader(
            word_nl_feature=in0,
            char_nl_feature=in1
        )
        assert (5, 1, 3) == out.data.shape
        assert (5, 1) == out.mask.shape

    def test_mask(self):
        reader = Encoder(2, 3, 1, 0.0, 5)
        in00 = torch.rand(5, 3)
        in01 = torch.rand(7, 3)
        in10 = torch.rand(5, 2)
        in11 = torch.rand(7, 2)
        out0 = reader(
            word_nl_feature=pad_sequence([in00, in01], 0),
            char_nl_feature=pad_sequence([in10, in11])
        )
        out1 = reader(
            word_nl_feature=pad_sequence([in00], 0),
            char_nl_feature=pad_sequence([in10])
        )
        out0 = out0.data[:5, :1, :]
        out1 = out1.data
        assert np.allclose(out0.detach().numpy(),
                           out1.detach().numpy())
