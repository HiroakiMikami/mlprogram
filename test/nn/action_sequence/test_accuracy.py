import numpy as np
import torch

from mlprogram import Environment
from mlprogram.nn.action_sequence import Accuracy
from mlprogram.nn.utils import rnn


class TestAccuracy(object):
    def test_parameters(self):
        acc = Accuracy()
        assert 0 == len(dict(acc.named_parameters()))

    def test_shape(self):
        gt0 = torch.LongTensor([[0, -1, -1], [-1, 2, -1], [-1, -1, 3]])
        gt = rnn.pad_sequence([gt0], padding_value=-1)
        rule_prob0 = torch.FloatTensor([[0.8, 0.2], [0.5, 0.5], [0.5, 0.5]])
        rule_prob = rnn.pad_sequence([rule_prob0])
        token_prob0 = torch.FloatTensor(
            [[0.1, 0.4, 0.5], [0.1, 0.2, 0.8], [0.5, 0.4, 0.1]])
        token_prob = rnn.pad_sequence([token_prob0])
        reference_prob0 = torch.FloatTensor(
            [[0.1, 0.4, 0.5, 0.0], [0.0, 0.5, 0.4, 0.1], [0.0, 0.0, 0.0, 1.0]])
        reference_prob = rnn.pad_sequence([reference_prob0])

        acc = Accuracy()
        a = acc(Environment(
            outputs={"rule_probs": rule_prob,
                     "token_probs": token_prob,
                     "reference_probs": reference_prob},
            supervisions={"ground_truth_actions": gt})
        ).outputs["action_sequence_accuracy"]
        assert () == a.shape

    def test_accuracy_if_match(self):
        gt0 = torch.LongTensor(
            [[0, -1, -1], [-1, 2, -1], [-1, -1, 3], [-1, 0, 0]])
        gt = rnn.pad_sequence([gt0], padding_value=-1)
        rule_prob0 = torch.FloatTensor([
            [0.8, 0.2], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        rule_prob = rnn.pad_sequence([rule_prob0])
        token_prob0 = torch.FloatTensor(
            [[0.1, 0.4, 0.5], [0.1, 0.2, 0.8], [0.5, 0.4, 0.1], [1.0, 0, 0]])
        token_prob = rnn.pad_sequence([token_prob0])
        reference_prob0 = torch.FloatTensor(
            [[0.1, 0.4, 0.5, 0.0], [0.0, 0.5, 0.4, 0.1], [0.0, 0.0, 0.0, 1.0],
             [1.0, 0.0, 0.0, 0.0]])
        reference_prob = rnn.pad_sequence([reference_prob0])

        acc = Accuracy()
        a = acc(Environment(
            outputs={"rule_probs": rule_prob,
                     "token_probs": token_prob,
                     "reference_probs": reference_prob},
            supervisions={"ground_truth_actions": gt})
        ).outputs["action_sequence_accuracy"]
        assert np.allclose(1.0, float(a.numpy()))
