import unittest
import torch

from mlprogram.nn import NL2ProgAccuracy
from mlprogram.nn.utils import rnn


class TestAccuracy(unittest.TestCase):
    def test_parameters(self):
        acc = NL2ProgAccuracy()
        self.assertEqual(0, len(dict(acc.named_parameters())))

    def test_shape(self):
        gt0 = torch.LongTensor([[0, -1, -1], [-1, 2, -1], [-1, -1, 3]])
        gt = rnn.pad_sequence([gt0], padding_value=-1)
        rule_prob0 = torch.FloatTensor([[0.8, 0.2], [0.5, 0.5], [0.5, 0.5]])
        rule_prob = rnn.pad_sequence([rule_prob0])
        token_prob0 = torch.FloatTensor(
            [[0.1, 0.4, 0.5], [0.1, 0.2, 0.8], [0.5, 0.4, 0.1]])
        token_prob = rnn.pad_sequence([token_prob0])
        copy_prob0 = torch.FloatTensor(
            [[0.1, 0.4, 0.5, 0.0], [0.0, 0.5, 0.4, 0.1], [0.0, 0.0, 0.0, 1.0]])
        copy_prob = rnn.pad_sequence([copy_prob0])

        acc = NL2ProgAccuracy()
        a = acc(rule_probs=rule_prob,
                token_probs=token_prob,
                copy_probs=copy_prob,
                ground_truth_actions=gt)
        self.assertEqual((), a.shape)

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
        copy_prob0 = torch.FloatTensor(
            [[0.1, 0.4, 0.5, 0.0], [0.0, 0.5, 0.4, 0.1], [0.0, 0.0, 0.0, 1.0],
             [1.0, 0.0, 0.0, 0.0]])
        copy_prob = rnn.pad_sequence([copy_prob0])

        acc = NL2ProgAccuracy()
        a = acc(rule_probs=rule_prob,
                token_probs=token_prob,
                copy_probs=copy_prob,
                ground_truth_actions=gt)
        self.assertAlmostEqual(1.0, float(a.numpy()))


if __name__ == "__main__":
    unittest.main()
