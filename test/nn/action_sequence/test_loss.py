import numpy as np
import torch

from mlprogram import Environment
from mlprogram.nn.action_sequence import Loss
from mlprogram.nn.utils import rnn


class TestLoss(object):
    def test_parameters(self):
        loss = Loss()
        assert 0 == len(dict(loss.named_parameters()))

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

        loss = Loss()
        objective = loss(Environment(
            {"rule_probs": rule_prob,
             "token_probs": token_prob,
             "reference_probs": reference_prob,
             "ground_truth_actions": gt},
            set(["ground_truth_actions"])
        ))["action_sequence_loss"]
        assert () == objective.shape

    def test_reduction(self):
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

        loss0 = Loss()
        loss1 = Loss(reduction="sum")
        loss2 = Loss(reduction="none")
        objective0 = loss0(Environment(
            {"rule_probs": rule_prob,
             "token_probs": token_prob,
             "reference_probs": reference_prob,
             "ground_truth_actions": gt},
            set(["ground_truth_actions"])
        ))["action_sequence_loss"]
        objective1 = loss1(Environment(
            {"rule_probs": rule_prob,
             "token_probs": token_prob,
             "reference_probs": reference_prob,
             "ground_truth_actions": gt},
            set(["ground_truth_actions"])
        ))["action_sequence_loss"]
        objective2 = loss2(Environment(
            {"rule_probs": rule_prob,
             "token_probs": token_prob,
             "reference_probs": reference_prob,
             "ground_truth_actions": gt},
            set(["ground_truth_actions"])
        ))["action_sequence_loss"]
        assert (1,) == objective2.shape
        assert np.allclose(objective0.item(), objective1.item())
