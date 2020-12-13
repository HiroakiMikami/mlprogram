import numpy as np
import torch

from mlprogram.builtins import Environment
from mlprogram.nn.nl2code import ActionSequenceReader, Predictor
from mlprogram.nn.utils import rnn


class TestPredictor(object):
    def test_parameters(self):
        reader = ActionSequenceReader(1, 1, 1, 2, 3)
        predictor = Predictor(reader, 1, 2, 3, 5)
        assert 17 == len(dict(predictor.named_parameters()))

    def test_shape(self):
        reader = ActionSequenceReader(1, 1, 1, 1, 1)
        predictor = Predictor(reader, 1, 2, 3, 5)
        feature0 = torch.rand(2, 3)
        feature1 = torch.rand(1, 3)
        feature = rnn.pad_sequence([feature0, feature1])
        context0 = torch.rand(2, 2)
        context1 = torch.rand(1, 2)
        context = rnn.pad_sequence([context0, context1])
        ref0 = torch.rand(3, 2)
        ref1 = torch.rand(1, 2)
        reference = rnn.pad_sequence([ref0, ref1])

        inputs = predictor(Environment({
            "reference_features": reference,
            "action_features": feature,
            "action_contexts": context
        }))
        rule_pred = inputs["rule_probs"]
        token_pred = inputs["token_probs"]
        reference_pred = inputs["reference_probs"]
        assert np.array_equal(
            [[1, 1], [1, 0]], rule_pred.mask.numpy())
        assert (2, 2, 1) == rule_pred.data.shape
        assert (2, 2, 1) == token_pred.data.shape
        assert np.array_equal(
            [[1, 1], [1, 0]], token_pred.mask.numpy())
        assert (2, 2, 3) == reference_pred.data.shape
        assert np.array_equal(
            [[1, 1], [1, 0]], reference_pred.mask.numpy())

    def test_shape_eval(self):
        reader = ActionSequenceReader(1, 1, 1, 1, 1)
        predictor = Predictor(reader, 1, 2, 3, 5)
        feature0 = torch.rand(2, 3)
        feature1 = torch.rand(1, 3)
        feature = rnn.pad_sequence([feature0, feature1])
        context0 = torch.rand(2, 2)
        context1 = torch.rand(1, 2)
        context = rnn.pad_sequence([context0, context1])
        ref0 = torch.rand(3, 2)
        ref1 = torch.rand(1, 2)
        reference = rnn.pad_sequence([ref0, ref1])

        predictor.eval()
        inputs = predictor(Environment({
            "reference_features": reference,
            "action_features": feature,
            "action_contexts": context
        }))
        rule_pred = inputs["rule_probs"]
        token_pred = inputs["token_probs"]
        reference_pred = inputs["reference_probs"]
        assert (2, 1) == rule_pred.shape
        assert (2, 1) == token_pred.shape
        assert (2, 3) == reference_pred.shape

    def test_probs(self):
        reader = ActionSequenceReader(1, 1, 1, 1, 1)
        predictor = Predictor(reader, 1, 2, 3, 5)
        feature0 = torch.rand(2, 3)
        feature1 = torch.rand(1, 3)
        feature = rnn.pad_sequence([feature0, feature1])
        context0 = torch.rand(2, 2)
        context1 = torch.rand(1, 2)
        context = rnn.pad_sequence([context0, context1])
        ref0 = torch.rand(3, 2)
        ref1 = torch.rand(1, 2)
        reference = rnn.pad_sequence([ref0, ref1])

        inputs = predictor(Environment({
            "reference_features": reference,
            "action_features": feature,
            "action_contexts": context
        }))
        rule_pred = inputs["rule_probs"].data
        token_pred = inputs["token_probs"].data
        reference_pred = inputs["reference_probs"].data
        probs = \
            torch.sum(rule_pred, dim=2) + torch.sum(token_pred, dim=2) + \
            torch.sum(reference_pred, dim=2)
        assert np.allclose([[1, 1], [1, 1]], probs.detach().numpy())
