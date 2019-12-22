import unittest
import torch
import numpy as np

from nl2prog.nn.pointer_net import PointerNet
from nl2prog.nn.utils.rnn import pad_sequence


class TestPointerNet(unittest.TestCase):
    def test_parameters(self):
        layer = PointerNet(1, 2, 3)
        params = dict(layer.named_parameters())
        self.assertEqual(6, len(params))
        self.assertEqual((3, 1), params["w1.weight"].shape)
        self.assertEqual((3, 2), params["w2.weight"].shape)
        self.assertEqual((1, 3), params["v.weight"].shape)

    def test_pointer_net(self):
        value0 = torch.FloatTensor([[1]])
        value1 = torch.FloatTensor([[1], [1], [1], [1]])
        value = pad_sequence([value0, value1])

        key0 = torch.FloatTensor([[0]])
        key1 = torch.FloatTensor([[0]])
        key = pad_sequence([key0, key1]).data

        layer = PointerNet(1, 1, 2)
        log_output = layer(key, value)  # (1, 2, 4)
        output = torch.exp(log_output)
        output *= value.mask.permute(1, 0).view(1, 2, 4).float()
        self.assertEqual((1, 2, 4), output.shape)
        self.assertTrue(np.allclose(
            [[1, 0, 0, 0], [0.25, 0.25, 0.25, 0.25]],
            output.detach().numpy()))


if __name__ == "__main__":
    unittest.main()
