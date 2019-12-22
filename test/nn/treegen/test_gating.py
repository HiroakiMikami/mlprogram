import unittest
import torch

from nl2prog.nn.treegen import Gating


class TestGating(unittest.TestCase):
    def test_parameters(self):
        gate = Gating(2, 3, 5, 7)
        pshape = {k: v.shape for k, v in gate.named_parameters()}
        self.assertEqual(5, len(list(gate.parameters())))
        self.assertEqual((5, 2), pshape["q.weight"])
        self.assertEqual((5, 2), pshape["w_k0.weight"])
        self.assertEqual((5, 3), pshape["w_k1.weight"])
        self.assertEqual((7, 2), pshape["w_f0.weight"])
        self.assertEqual((7, 3), pshape["w_f1.weight"])

    def test_shape(self):
        gate = Gating(2, 3, 5, 7)
        in0 = torch.Tensor(11, 13, 2)
        in1 = torch.Tensor(11, 13, 3)
        out = gate(in0, in1)
        self.assertEqual((11, 13, 7), out.shape)


if __name__ == "__main__":
    unittest.main()
