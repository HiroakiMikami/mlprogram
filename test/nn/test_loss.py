import unittest
import torch

from nl2code.nn import Loss
from nl2code.nn.utils import rnn


class TestLoss(unittest.TestCase):
    def test_parameters(self):
        loss = Loss()
        self.assertEqual(0, len(dict(loss.named_parameters())))

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

        loss = Loss()
        objective = loss(rule_prob, token_prob, copy_prob, gt)
        self.assertEqual((), objective.shape)


if __name__ == "__main__":
    unittest.main()
