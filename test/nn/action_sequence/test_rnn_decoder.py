import unittest
import torch
import numpy as np

from mlprogram.nn.action_sequence import RnnDecoder
from mlprogram.nn.utils import rnn


class TestRnnDecoder(unittest.TestCase):
    def test_parameters(self):
        decoder = RnnDecoder(2, 3, 5, 0.0)
        self.assertEqual(4, len(dict(decoder.named_parameters())))

    def test_shape(self):
        decoder = RnnDecoder(2, 3, 5, 0.0)
        input = torch.rand(2, 2)
        action0 = torch.rand(3, 3)  # length = 3
        action1 = torch.rand(1, 3)  # length = 1
        action = rnn.pad_sequence([action0, action1])
        h_0 = torch.rand(2, 5)
        c_0 = torch.rand(2, 5)

        inputs = decoder({
            "input_feature": input,
            "action_features": action,
            "hidden_state": h_0,
            "state": c_0
        })
        output = inputs["action_features"]
        h_n = inputs["hidden_state"]
        c_n = inputs["state"]
        self.assertEqual((3, 2, 5), output.data.shape)
        self.assertTrue(np.array_equal(
            [[1, 1], [1, 0], [1, 0]], output.mask.numpy()))
        self.assertEqual((2, 5), h_n.shape)
        self.assertEqual((2, 5), c_n.shape)


if __name__ == "__main__":
    unittest.main()
