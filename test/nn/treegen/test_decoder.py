import torch
import numpy as np
import unittest

from nl2prog.nn.treegen import DecoderBlock, Decoder
from nl2prog.nn.utils.rnn import pad_sequence


class TestDecoderBlock(unittest.TestCase):
    def test_parameters(self):
        block = DecoderBlock(1, 3, 5, 1, 0.0)
        self.assertEqual(18, len(list(block.parameters())))

    def test_shape(self):
        block = DecoderBlock(1, 3, 5, 1, 0.0)
        query0 = torch.Tensor(7, 1)
        nl0 = torch.Tensor(11, 1)
        ast0 = torch.Tensor(7, 1)
        out, w0, w1 = block(pad_sequence([query0], 0),
                            pad_sequence([nl0], 0),
                            pad_sequence([ast0], 0))
        self.assertEqual((7, 1, 5), out.data.shape)
        self.assertEqual((7, 1), out.mask.shape)
        self.assertEqual((1, 7, 11), w0.shape)
        self.assertEqual((1, 7, 7), w1.shape)

    def test_mask_nl(self):
        block = DecoderBlock(1, 3, 5, 1, 0.0)
        query0 = torch.rand(7, 1)
        nl0 = torch.rand(11, 1)
        nl1 = torch.rand(13, 1)
        ast0 = torch.rand(7, 1)
        out0, w00, w01 = block(pad_sequence([query0], 0),
                               pad_sequence([nl0], 0),
                               pad_sequence([ast0], 0))
        out1, w10, w11 = block(pad_sequence([query0, query0], 0),
                               pad_sequence([nl0, nl1], 0),
                               pad_sequence([ast0, ast0], 0))
        out0 = out0.data
        out1 = out1.data[:7, :1, :]
        w10 = w10[:1, :7, :11]
        w11 = w11[:1, :7, :7]
        self.assertTrue(np.allclose(out0.detach().numpy(),
                                    out1.detach().numpy()))
        self.assertTrue(np.allclose(w00.detach().numpy(),
                                    w10.detach().numpy()))
        self.assertTrue(np.allclose(w01.detach().numpy(),
                                    w11.detach().numpy()))

    def test_mask_ast_and_query(self):
        block = DecoderBlock(1, 3, 5, 1, 0.0)
        query0 = torch.rand(7, 1)
        query1 = torch.rand(9, 1)
        nl0 = torch.rand(11, 1)
        ast0 = torch.rand(7, 1)
        ast1 = torch.rand(9, 1)
        out0, w00, w01 = block(pad_sequence([query0], 0),
                               pad_sequence([nl0], 0),
                               pad_sequence([ast0], 0))
        out1, w10, w11 = block(pad_sequence([query0, query1], 0),
                               pad_sequence([nl0, nl0], 0),
                               pad_sequence([ast0, ast1], 0))
        out0 = out0.data
        out1 = out1.data[:7, :1, :]
        w10 = w10[:1, :7, :11]
        w11 = w11[:1, :7, :7]
        self.assertTrue(np.allclose(out0.detach().numpy(),
                                    out1.detach().numpy()))
        self.assertTrue(np.allclose(w00.detach().numpy(),
                                    w10.detach().numpy()))
        self.assertTrue(np.allclose(w01.detach().numpy(),
                                    w11.detach().numpy()))

    def test_attn_mask(self):
        block = DecoderBlock(1, 3, 5, 1, 0.0)
        query0 = torch.rand(7, 1)
        nl0 = torch.rand(11, 1)
        ast0 = torch.rand(7, 1)
        out0, w00, w01 = block(pad_sequence([query0[:5, :]], 0),
                               pad_sequence([nl0], 0),
                               pad_sequence([ast0[:5, :]], 0))
        out1, w10, w11 = block(pad_sequence([query0], 0),
                               pad_sequence([nl0], 0),
                               pad_sequence([ast0], 0))
        out0 = out0.data
        out1 = out1.data[:5, :1, :]
        w10 = w10[:1, :5, :11]
        w11 = w11[:1, :5, :5]
        self.assertTrue(np.allclose(out0.detach().numpy(),
                                    out1.detach().numpy()))
        self.assertTrue(np.allclose(w00.detach().numpy(),
                                    w10.detach().numpy()))
        self.assertTrue(np.allclose(w01.detach().numpy(),
                                    w11.detach().numpy()))


class TestDecoder(unittest.TestCase):
    def test_parameters(self):
        decoder = Decoder(1, 1, 3, 1, 3, 5, 1, 0.0, 5)
        self.assertEqual(18 * 5 + 2, len(list(decoder.parameters())))

    def test_shape(self):
        decoder = Decoder(1, 1, 3, 1, 3, 5, 1, 0.0, 5)
        query0 = torch.zeros(7, 3).long()
        nl0 = torch.Tensor(11, 1)
        ast0 = torch.Tensor(7, 1)
        out, _ = decoder(pad_sequence([query0], 0),
                         pad_sequence([nl0], 0), None,
                         pad_sequence([ast0], 0))
        self.assertEqual((7, 1, 5), out.data.shape)
        self.assertEqual((7, 1), out.mask.shape)

    def test_mask_nl(self):
        decoder = Decoder(1, 1, 3, 1, 3, 5, 1, 0.0, 5)
        query0 = torch.zeros(7, 3).long()
        nl0 = torch.rand(11, 1)
        nl1 = torch.rand(13, 1)
        ast0 = torch.rand(7, 1)
        out0, _ = decoder(pad_sequence([query0], 0),
                          pad_sequence([nl0], 0), None,
                          pad_sequence([ast0], 0))
        out1, _ = decoder(pad_sequence([query0, query0], 0),
                          pad_sequence([nl0, nl1], 0), None,
                          pad_sequence([ast0, ast0], 0))
        out0 = out0.data
        out1 = out1.data[:7, :1, :]
        self.assertTrue(np.allclose(out0.detach().numpy(),
                                    out1.detach().numpy()))

    def test_mask_ast_and_query(self):
        block = Decoder(1, 1, 3, 1, 3, 5, 1, 0.0, 5)
        query0 = torch.zeros(7, 3).long()
        query1 = torch.zeros(9, 3).long()
        nl0 = torch.rand(11, 1)
        ast0 = torch.rand(7, 1)
        ast1 = torch.rand(9, 1)
        out0, _ = block(pad_sequence([query0], 0),
                        pad_sequence([nl0], 0), None,
                        pad_sequence([ast0], 0))
        out1, _ = block(pad_sequence([query0, query1], 0),
                        pad_sequence([nl0, nl0], 0), None,
                        pad_sequence([ast0, ast1], 0))
        out0 = out0.data
        out1 = out1.data[:7, :1, :]
        self.assertTrue(np.allclose(out0.detach().numpy(),
                                    out1.detach().numpy()))

    def test_attn_mask(self):
        block = Decoder(1, 1, 3, 1, 3, 5, 1, 0.0, 5)
        query0 = torch.zeros(7, 3).long()
        nl0 = torch.rand(11, 1)
        ast0 = torch.rand(7, 1)
        out0, _ = block(pad_sequence([query0[:5, :]], 0),
                        pad_sequence([nl0], 0), None,
                        pad_sequence([ast0[:5, :]], 0))
        out1, _ = block(pad_sequence([query0], 0),
                        pad_sequence([nl0], 0), None,
                        pad_sequence([ast0], 0))
        out0 = out0.data
        out1 = out1.data[:5, :1, :]
        self.assertTrue(np.allclose(out0.detach().numpy(),
                                    out1.detach().numpy()))


if __name__ == "__main__":
    unittest.main()
