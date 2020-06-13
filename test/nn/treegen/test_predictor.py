import unittest
import torch
import numpy as np

from mlprogram.nn.treegen import Predictor
from mlprogram.nn.utils.rnn import pad_sequence


class TestPredictor(unittest.TestCase):
    def test_parameters(self):
        predictor = Predictor(2, 3, 5, 7, 11)
        pshape = {k: v.shape for k, v in predictor.named_parameters()}
        self.assertEqual(12, len(list(predictor.parameters())))
        self.assertEqual((3, 2), pshape["select.weight"])
        self.assertEqual((3,), pshape["select.bias"])
        self.assertEqual((5, 2), pshape["rule.weight"])
        self.assertEqual((5,), pshape["rule.bias"])
        self.assertEqual((7, 2), pshape["token.weight"])
        self.assertEqual((7,), pshape["token.bias"])
        self.assertEqual((11, 2), pshape["copy.w1.weight"])
        self.assertEqual((11,), pshape["copy.w1.bias"])
        self.assertEqual((11, 3), pshape["copy.w2.weight"])
        self.assertEqual((11,), pshape["copy.w2.bias"])
        self.assertEqual((1, 11), pshape["copy.v.weight"])
        self.assertEqual((1,), pshape["copy.v.bias"])

    def test_shape(self):
        predictor = Predictor(2, 3, 5, 7, 11)
        f = torch.Tensor(11, 2)
        nl = torch.Tensor(13, 3)
        inputs = predictor(nl_query_features=pad_sequence([nl]),
                           action_features=pad_sequence([f]))
        rule = inputs["rule_probs"]
        token = inputs["token_probs"]
        copy = inputs["copy_probs"]
        self.assertEqual((11, 1, 5), rule.data.shape)
        self.assertEqual((11, 1), rule.mask.shape)
        self.assertEqual((11, 1, 7), token.data.shape)
        self.assertEqual((11, 1), token.mask.shape)
        self.assertEqual((11, 1, 13), copy.data.shape)
        self.assertEqual((11, 1), copy.mask.shape)

    def test_prog(self):
        predictor = Predictor(2, 3, 5, 7, 11)
        f = torch.rand(11, 2)
        nl = torch.rand(13, 3)
        inputs = predictor(nl_query_features=pad_sequence([nl]),
                           action_features=pad_sequence([f]))
        rule = inputs["rule_probs"]
        token = inputs["token_probs"]
        copy = inputs["copy_probs"]
        prob = torch.cat([rule.data, token.data, copy.data], dim=2)
        prob = prob.detach().numpy()
        self.assertTrue(np.all(prob >= 0.0))
        self.assertTrue(np.all(prob <= 1.0))
        total = np.sum(prob, axis=2)
        self.assertTrue(np.allclose(1.0, total))

    def test_nl_mask(self):
        predictor = Predictor(2, 3, 5, 7, 11)
        f0 = torch.rand(11, 2)
        f1 = torch.rand(13, 2)
        nl0 = torch.rand(13, 3)
        nl1 = torch.rand(15, 3)
        inputs0 = predictor(nl_query_features=pad_sequence([nl0]),
                            action_features=pad_sequence([f0]))
        rule0 = inputs0["rule_probs"]
        token0 = inputs0["token_probs"]
        copy0 = inputs0["copy_probs"]
        inputs1 = predictor(nl_query_features=pad_sequence([nl0, nl1]),
                            action_features=pad_sequence([f0, f1]))
        rule1 = inputs1["rule_probs"]
        token1 = inputs1["token_probs"]
        copy1 = inputs1["copy_probs"]
        rule1 = rule1.data[:11, :1, :]
        token1 = token1.data[:11, :1, :]
        copy1 = copy1.data[:11, :1, :13]
        self.assertTrue(np.allclose(rule0.data.detach().numpy(),
                                    rule1.detach().numpy()))
        self.assertTrue(np.allclose(token0.data.detach().numpy(),
                                    token1.detach().numpy()))
        self.assertTrue(np.allclose(copy0.data.detach().numpy(),
                                    copy1.detach().numpy()))


if __name__ == "__main__":
    unittest.main()
