import unittest
import torch
import numpy as np

from nl2prog.nn.treegen import PointerNetwork


class TestPointerNetwork(unittest.TestCase):
    def test_parameters(self):
        pnet = PointerNetwork(2, 3, 5)
        pshape = {k: v.shape for k, v in pnet.named_parameters()}
        self.assertEqual(3, len(list(pnet.parameters())))
        self.assertEqual((5, 2), pshape["w1.weight"])
        self.assertEqual((5, 3), pshape["w2.weight"])
        self.assertEqual((1, 5), pshape["v.weight"])

    def test_shape(self):
        pnet = PointerNetwork(2, 3, 5)
        key = torch.Tensor(7, 2)
        value = torch.Tensor(11, 7, 3)
        value_mask = torch.ones(11, 7)
        log_prob = pnet(key, value, value_mask)
        self.assertEqual((11, 7), log_prob.shape)

    def test_output_value(self):
        pnet = PointerNetwork(2, 3, 5)
        key = torch.rand(7, 2)
        value = torch.rand(11, 7, 3)
        value_mask = torch.ones(11, 7)
        log_prob = pnet(key, value, value_mask)
        prob = torch.exp(log_prob)
        total = prob.sum(dim=0)
        self.assertEqual((7,), total.shape)
        self.assertTrue(np.allclose(np.ones((7,)), total.detach().numpy()))

    def test_mask(self):
        pnet = PointerNetwork(2, 3, 5)
        key = torch.rand(1, 2)
        value = torch.rand(2, 1, 3)
        value_mask = torch.tensor([[1], [0]])
        log_prob = pnet(key, value, value_mask)
        prob = torch.exp(log_prob) * value_mask.float()
        self.assertTrue(np.allclose([[1], [0]], prob.detach().numpy()))


if __name__ == "__main__":
    unittest.main()
