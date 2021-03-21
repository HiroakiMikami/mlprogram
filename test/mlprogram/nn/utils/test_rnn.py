import numpy as np
import torch
import torch.nn.utils as utils

import mlprogram.nn.utils.rnn as rnn


class TestPadPackedSequence(object):
    def test_one_tensor(self):
        x = torch.FloatTensor([1, 1])
        packed = utils.rnn.pack_sequence([x])
        result = rnn.pad_packed_sequence(packed)
        assert np.allclose([[1], [1]], result.data.numpy())
        assert np.array_equal([[1], [1]], result.mask.numpy())

        x = torch.FloatTensor([[1, 2], [3, 4]])
        packed = utils.rnn.pack_sequence([x])
        result = rnn.pad_packed_sequence(packed)
        assert np.allclose([[[1, 2]], [[3, 4]]], result.data.numpy())
        assert np.array_equal([[1], [1]], result.mask.numpy())

    def test_multiple_tensors(self):
        x1 = torch.FloatTensor([1])
        x2 = torch.FloatTensor([2, 2])
        packed = utils.rnn.pack_sequence([x1, x2], enforce_sorted=False)
        result = rnn.pad_packed_sequence(packed)
        assert np.allclose([[1, 2], [0, 2]], result.data.numpy())
        assert np.array_equal([[1, 1], [0, 1]], result.mask.numpy())

    def test_padding_value(self):
        x1 = torch.FloatTensor([1])
        x2 = torch.FloatTensor([2, 2])
        packed = utils.rnn.pack_sequence([x1, x2], enforce_sorted=False)
        result = rnn.pad_packed_sequence(packed, -1)
        assert np.allclose([[1, 2], [-1, 2]], result.data.numpy())
        assert np.array_equal([[1, 1], [0, 1]], result.mask.numpy())


class TestPadSequence(object):
    def test_one_tensor(self):
        x = torch.FloatTensor([1, 1])
        result = rnn.pad_sequence([x])
        assert np.allclose([[1], [1]], result.data.numpy())
        assert np.array_equal([[1], [1]], result.mask.numpy())

    def test_empty(self):
        x1 = torch.zeros((0, 1))
        x2 = torch.FloatTensor([[2], [2]])
        result = rnn.pad_sequence([x1, x2])
        assert np.allclose([[[0], [2]], [[0], [2]]],
                           result.data.numpy())
        assert np.array_equal([[0, 1], [0, 1]], result.mask.numpy())
