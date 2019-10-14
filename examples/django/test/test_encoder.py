import torch
import torch.nn.utils.rnn as rnn
import unittest
from examples.django import Encoder


class TestEncoder(unittest.TestCase):
    def test_parameters(self):
        encoder = Encoder(5, 3, 14)
        self.assertEqual(9, len(list(encoder.named_parameters())))

    def test_shape(self):
        encoder = Encoder(5, 3, 14, 2)
        q0 = torch.LongTensor([1, 2])
        q1 = torch.LongTensor([1, 2, 3])
        query = rnn.pack_sequence([q0, q1], enforce_sorted=False)
        output, (h_n, c_n) = encoder(query)
        self.assertEqual((3, 2, 14),
                         rnn.pad_packed_sequence(output)[0].data.shape)
        self.assertEqual((4, 2, 7), h_n.shape)
        self.assertEqual((4, 2, 7), c_n.shape)


if __name__ == "__main__":
    unittest.main()
