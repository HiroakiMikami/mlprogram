import unittest
import torch
import numpy as np

from nl2prog.nn.nl2code.decoder import query_history
from nl2prog.nn.nl2code import DecoderCell, Decoder
from nl2prog.nn.utils import rnn


class TestQueryHistory(unittest.TestCase):
    def test_simple_case(self):
        # length = 3, minibatch-size = 2
        history = torch.FloatTensor([[[1], [-1]], [[2], [-2]], [[3], [-3]]])
        index = torch.LongTensor([0, 2])
        h = query_history(history, index)
        self.assertEqual((2, 1), h.shape)
        self.assertTrue(np.array_equal([[1], [-3]], h.numpy()))


class TestDecoderCell(unittest.TestCase):
    def test_parameters(self):
        cell = DecoderCell(2, 3, 5, 7, 0.0)
        self.assertEqual(8, len(dict(cell.named_parameters())))

    def test_shape(self):
        cell = DecoderCell(2, 3, 5, 7, 0.0)
        ctx0 = torch.rand(3, 2)
        ctx1 = torch.rand(3, 1)
        ctx = rnn.pad_sequence([ctx0, ctx1])
        input = torch.rand(2, 3)
        parent_index = torch.randint(10, size=(2,))
        history = torch.rand(10, 2, 5)
        h_0 = torch.rand(2, 5)
        c_0 = torch.rand(2, 5)

        ctx, (h_1, c_1) = cell(ctx, input, parent_index, history, (h_0, c_0))
        self.assertEqual((2, 2), ctx.shape)
        self.assertEqual((2, 5), h_1.shape)
        self.assertEqual((2, 5), c_1.shape)


class TestDecoder(unittest.TestCase):
    def test_parameters(self):
        decoder = Decoder(2, 3, 5, 7, 0.0)
        self.assertEqual(8, len(dict(decoder.named_parameters())))

    def test_shape(self):
        decoder = Decoder(2, 3, 5, 7, 0.0)
        query0 = torch.rand(3, 2)
        query1 = torch.rand(1, 2)
        query = rnn.pad_sequence([query0, query1])
        input0 = torch.rand(3, 3)  # length = 3
        input1 = torch.rand(1, 3)  # length = 1
        input = rnn.pad_sequence([input0, input1])
        parent_index0 = torch.randint(10, size=(3,))
        parent_index1 = torch.randint(10, size=(1,))
        parent_index = rnn.pad_sequence([parent_index0, parent_index1])
        history = torch.rand(10, 2, 5)
        h_0 = torch.rand(2, 5)
        c_0 = torch.rand(2, 5)

        output, contexts, history, (h_n, c_n) = decoder(
            query, input, parent_index, history, (h_0, c_0))
        self.assertEqual((3, 2, 5), output.data.shape)
        self.assertTrue(np.array_equal(
            [[1, 1], [1, 0], [1, 0]], output.mask.numpy()))
        self.assertEqual((3, 2, 2), contexts.data.shape)
        self.assertTrue(np.array_equal(
            [[1, 1], [1, 0], [1, 0]], contexts.mask.numpy()))
        self.assertEqual((13, 2, 5), history.shape)
        self.assertEqual((2, 5), h_n.shape)
        self.assertEqual((2, 5), c_n.shape)


if __name__ == "__main__":
    unittest.main()
