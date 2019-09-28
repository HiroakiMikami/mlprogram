import unittest
import torch
import numpy as np

from nl2code.nn._pointer_net import PointerNet
from nl2code.nn.utils.rnn import pad_sequence


class TestPointerNet(unittest.TestCase):
    def test_parameters(self):
        layer = PointerNet(1, 2, 3)
        params = dict(layer.named_parameters())
        self.assertEqual(6, len(params))
        self.assertEqual((3, 1), params["_l1_q.weight"].shape)
        self.assertEqual((3, 2), params["_l1_h.weight"].shape)
        self.assertEqual((1, 3), params["_l2.weight"].shape)

    def test_pointer_net(self):
        query0 = torch.FloatTensor([[1]])
        query1 = torch.FloatTensor([[1], [1], [1], [1]])
        query = pad_sequence([query0, query1])

        decoder_state0 = torch.FloatTensor([[0]])
        decoder_state1 = torch.FloatTensor([[0]])
        decoder_state = pad_sequence([decoder_state0, decoder_state1])

        layer = PointerNet(1, 1, 2)
        output = layer(query, decoder_state)  # (1, 2, 4)
        self.assertEqual((1, 2, 4), output.data.shape)
        self.assertTrue(np.array_equal([[1, 1]], output.mask.detach().numpy()))
        self.assertTrue(np.allclose(
            [[1, 0, 0, 0], [0.25, 0.25, 0.25, 0.25]],
            output.data.detach().numpy()))


if __name__ == "__main__":
    unittest.main()
