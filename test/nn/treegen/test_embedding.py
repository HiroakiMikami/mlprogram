import unittest
import numpy as np
import torch

from nl2prog.nn.treegen import Embedding


class TestEmbedding(unittest.TestCase):
    def test_parameters(self):
        e = Embedding(1, 2, 3, 5, 7, 11)
        pshape = {k: v.shape for k, v in e.named_parameters()}
        self.assertEqual(3, len(list(e.parameters())))
        self.assertEqual((1, 5), pshape["embed.weight"])
        self.assertEqual((3, 7), pshape["elem_embed.weight"])
        self.assertEqual((11, 7, 3), pshape["elem_to_seq.weight"])

    def test_shape(self):
        e = Embedding(1, 2, 3, 5, 7, 11)
        in0 = torch.zeros(13, 1, dtype=torch.long)
        in1 = torch.zeros(13, 1, 3, dtype=torch.long)
        out0, out1 = e(in0, in1)
        self.assertEqual((13, 1, 5), out0.shape)
        self.assertEqual((13, 1, 11), out1.shape)

    def test_mask(self):
        e0 = Embedding(1, 2, 3, 5, 7, 11)
        e1 = Embedding(1, 2, 2, 5, 7, 11)
        e0.embed.weight.data = e1.embed.weight.data
        e0.elem_embed.weight.data = e1.elem_embed.weight.data
        e0.elem_to_seq.weight.data = e1.elem_to_seq.weight.data[:, :, :2]

        in0 = torch.zeros(11, 1, dtype=torch.long)
        in1 = torch.randint(3, [11, 1, 3])
        in1[:, :, 2] = 0
        with torch.no_grad():
            _, out0 = e0(in0, in1)
            _, out1 = e1(in0, in1)
        self.assertTrue(np.allclose(out0.numpy(), out1.numpy()))


if __name__ == "__main__":
    unittest.main()
