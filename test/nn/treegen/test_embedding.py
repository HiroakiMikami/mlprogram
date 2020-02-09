import unittest
import numpy as np
import torch

from nl2prog.nn.treegen import QueryEmbedding, RuleEmbedding


class TestQueryEmbedding(unittest.TestCase):
    def test_parameters(self):
        e = QueryEmbedding(1, 2, 3, 5, 7, 11)
        pshape = {k: v.shape for k, v in e.named_parameters()}
        self.assertEqual(3, len(list(e.parameters())))
        self.assertEqual((1, 5), pshape["embed.weight"])
        self.assertEqual((3, 7), pshape["elem_embed.weight"])
        self.assertEqual((11, 7, 3), pshape["elem_to_seq.weight"])

    def test_shape(self):
        e = QueryEmbedding(1, 2, 3, 5, 7, 11)
        in0 = torch.zeros(13, 1, dtype=torch.long)
        in1 = torch.zeros(13, 1, 3, dtype=torch.long)
        out0, out1 = e(in0, in1)
        self.assertEqual((13, 1, 5), out0.shape)
        self.assertEqual((13, 1, 11), out1.shape)

    def test_mask(self):
        e0 = QueryEmbedding(1, 2, 3, 5, 7, 11)
        e1 = QueryEmbedding(1, 2, 2, 5, 7, 11)
        e1.embed.weight.data = e0.embed.weight.data
        e1.elem_embed.weight.data = e0.elem_embed.weight.data
        e1.elem_to_seq.weight.data = e0.elem_to_seq.weight.data[:, :, :2]

        in0 = torch.zeros(11, 1, dtype=torch.long)
        in1 = torch.randint(3, [11, 1, 3])
        in1[:, :, 2] = -1
        with torch.no_grad():
            _, out0 = e0(in0, in1)
            _, out1 = e1(in0, in1[:, :, :2])
        self.assertTrue(np.allclose(out0.numpy(), out1.numpy()))


class TestRuleEmbedding(unittest.TestCase):
    def test_parameters(self):
        e = RuleEmbedding(1, 2, 3, 5, 7, 11, 13)
        pshape = {k: v.shape for k, v in e.named_parameters()}
        self.assertEqual(5, len(list(e.parameters())))
        self.assertEqual((2, 7), pshape["rule_embed.weight"])
        self.assertEqual((4, 7), pshape["token_embed.weight"])
        self.assertEqual((4, 11), pshape["elem_node_type_embed.weight"])
        self.assertEqual((4, 11), pshape["elem_token_embed.weight"])
        self.assertEqual((13, 11, 6), pshape["elem_to_seq.weight"])

    def test_shape(self):
        e = RuleEmbedding(1, 2, 3, 5, 7, 11, 13)
        in0 = torch.zeros(13, 1, 3, dtype=torch.long)
        in1 = torch.zeros(13, 1, 6, 3, dtype=torch.long)
        out0, out1 = e(in0, in1)
        self.assertEqual((13, 1, 7), out0.shape)
        self.assertEqual((13, 1, 13), out1.shape)

    def test_mask(self):
        e0 = RuleEmbedding(1, 2, 3, 5, 7, 11, 13)
        e1 = RuleEmbedding(1, 2, 3, 6, 7, 11, 13)
        e0.rule_embed.weight.data = e1.rule_embed.weight.data
        e0.token_embed.weight.data = e1.token_embed.weight.data
        e0.elem_node_type_embed.weight.data = \
            e1.elem_node_type_embed.weight.data
        e0.elem_token_embed.weight.data = e1.elem_token_embed.weight.data
        e0.elem_to_seq.weight.data = e1.elem_to_seq.weight.data[:, :, :6]

        in0 = torch.zeros(11, 1, 3, dtype=torch.long)
        in1 = torch.zeros(11, 1, 7, 3, dtype=torch.long)
        in1[:, :, 6, :] = -1
        with torch.no_grad():
            _, out0 = e0(in0, in1[:, :, :6, :])
            _, out1 = e1(in0, in1)
        self.assertTrue(np.allclose(out0.numpy(), out1.numpy()))

    def test_rule_mask(self):
        e = RuleEmbedding(1, 2, 3, 5, 7, 11, 13)
        in0 = torch.zeros(13, 1, 3, dtype=torch.long)
        in1 = torch.zeros(13, 1, 6, 3, dtype=torch.long)
        in0[:, :, 0] = -1
        out0, out1 = e(in0, in1)

    def test_token_mask(self):
        e = RuleEmbedding(1, 2, 3, 5, 7, 11, 13)
        in0 = torch.zeros(13, 1, 3, dtype=torch.long)
        in1 = torch.zeros(13, 1, 6, 3, dtype=torch.long)
        in0[:, :, 1] = -1
        in1[:, :, 1] = -1
        out0, out1 = e(in0, in1)

    def test_node_type_mask(self):
        e = RuleEmbedding(1, 2, 3, 5, 7, 11, 13)
        in0 = torch.zeros(13, 1, 3, dtype=torch.long)
        in1 = torch.zeros(13, 1, 6, 3, dtype=torch.long)
        in1[:, :, 0] = -1
        out0, out1 = e(in0, in1)

    def test_copy_mask(self):
        e = RuleEmbedding(1, 2, 3, 5, 7, 11, 13)
        in0 = torch.zeros(13, 1, 3, dtype=torch.long)
        in1 = torch.zeros(13, 1, 6, 3, dtype=torch.long)
        in0[:, :, 2] = -1
        in1[:, :, 2] = -1
        out0, out1 = e(in0, in1)

    def test_copy_embed(self):
        e = RuleEmbedding(1, 2, 3, 5, 7, 11, 13)
        in0 = torch.ones(2, 1, 3, dtype=torch.long) * -1
        in1 = torch.ones(2, 1, 6, 3, dtype=torch.long) * -1
        in0[0, :, 2] = 0
        in0[1, :, 2] = 1
        in1[0, :, :, 2] = 0
        in1[1, :, :, 2] = 1
        with torch.no_grad():
            out0, out1 = e(in0, in1)

        self.assertTrue(np.allclose(out0[0].numpy(), out0[1].numpy()))
        self.assertTrue(np.allclose(out1[0].numpy(), out1[1].numpy()))


if __name__ == "__main__":
    unittest.main()
