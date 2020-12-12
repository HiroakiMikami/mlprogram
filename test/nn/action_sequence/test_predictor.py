import numpy as np
import torch

from mlprogram import Environment
from mlprogram.nn.action_sequence import Predictor
from mlprogram.nn.utils.rnn import pad_sequence


class TestPredictor(object):
    def test_parameters(self):
        predictor = Predictor(2, 3, 5, 7, 11)
        pshape = {k: v.shape for k, v in predictor.named_parameters()}
        assert 12 == len(list(predictor.parameters()))
        assert (3, 2) == pshape["select.weight"]
        assert (3,) == pshape["select.bias"]
        assert (5, 2) == pshape["rule.weight"]
        assert (5,) == pshape["rule.bias"]
        assert (7, 2) == pshape["token.weight"]
        assert (7,) == pshape["token.bias"]
        assert (11, 2) == pshape["reference.w1.weight"]
        assert (11,) == pshape["reference.w1.bias"]
        assert (11, 3) == pshape["reference.w2.weight"]
        assert (11,) == pshape["reference.w2.bias"]
        assert (1, 11) == pshape["reference.v.weight"]
        assert (1,) == pshape["reference.v.bias"]

    def test_shape(self):
        predictor = Predictor(2, 3, 5, 7, 11)
        f = torch.Tensor(11, 2)
        nl = torch.Tensor(13, 3)
        inputs = predictor(Environment(
            {"reference_features": pad_sequence([nl]),
             "action_features": pad_sequence([f])}))
        rule = inputs["rule_probs"]
        token = inputs["token_probs"]
        reference = inputs["reference_probs"]
        assert (11, 1, 5) == rule.data.shape
        assert (11, 1) == rule.mask.shape
        assert (11, 1, 7) == token.data.shape
        assert (11, 1) == token.mask.shape
        assert (11, 1, 13) == reference.data.shape
        assert (11, 1) == reference.mask.shape

    def test_shape_eval(self):
        predictor = Predictor(2, 3, 5, 7, 11)
        f = torch.Tensor(11, 2)
        nl = torch.Tensor(13, 3)
        predictor.eval()
        inputs = predictor(Environment(
            {"reference_features": pad_sequence([nl]),
             "action_features": pad_sequence([f])}))
        rule = inputs["rule_probs"]
        token = inputs["token_probs"]
        reference = inputs["reference_probs"]
        assert (1, 5) == rule.shape
        assert (1, 7) == token.shape
        assert (1, 13) == reference.shape

    def test_prog(self):
        predictor = Predictor(2, 3, 5, 7, 11)
        f = torch.rand(11, 2)
        nl = torch.rand(13, 3)
        inputs = predictor(Environment(
            {"reference_features": pad_sequence([nl]),
             "action_features": pad_sequence([f])}))
        rule = inputs["rule_probs"]
        token = inputs["token_probs"]
        reference = inputs["reference_probs"]
        prob = torch.cat([rule.data, token.data, reference.data], dim=2)
        prob = prob.detach().numpy()
        assert np.all(prob >= 0.0)
        assert np.all(prob <= 1.0)
        total = np.sum(prob, axis=2)
        assert np.allclose(1.0, total)

    def test_nl_mask(self):
        predictor = Predictor(2, 3, 5, 7, 11)
        f0 = torch.rand(11, 2)
        f1 = torch.rand(13, 2)
        nl0 = torch.rand(13, 3)
        nl1 = torch.rand(15, 3)
        inputs0 = predictor(Environment(
            {"reference_features": pad_sequence([nl0]),
             "action_features": pad_sequence([f0])}))
        rule0 = inputs0["rule_probs"]
        token0 = inputs0["token_probs"]
        ref0 = inputs0["reference_probs"]
        inputs1 = predictor(Environment(
            {"reference_features": pad_sequence([nl0, nl1]),
             "action_features": pad_sequence([f0, f1])}))
        rule1 = inputs1["rule_probs"]
        token1 = inputs1["token_probs"]
        ref1 = inputs1["reference_probs"]
        rule1 = rule1.data[:11, :1, :]
        token1 = token1.data[:11, :1, :]
        ref1 = ref1.data[:11, :1, :13]
        assert np.allclose(rule0.data.detach().numpy(),
                           rule1.detach().numpy())
        assert np.allclose(token0.data.detach().numpy(),
                           token1.detach().numpy())
        assert np.allclose(ref0.data.detach().numpy(),
                           ref1.detach().numpy())
