import unittest
import torch
import numpy as np

from mlprogram.nn.action_sequence import ActionSequenceReader
from mlprogram.nn.utils import rnn


class TestActionSequenceReader(unittest.TestCase):
    def test_parameters(self):
        reader = ActionSequenceReader(2, 3, 5)
        self.assertEqual(2, len(dict(reader.named_parameters())))

    def test_shape(self):
        """
        num_rules = 2
        num_tokens = 3
        """
        reader = ActionSequenceReader(2, 3, 5)
        prev_action0 = torch.LongTensor([[0, 0, 0]])
        prev_action1 = torch.LongTensor([[0, 0, 0], [1, 1, 1]])
        prev_action = rnn.pad_sequence(
            [prev_action0, prev_action1])  # (2, 2, 3)

        data = reader({"previous_actions": prev_action})
        feature = data["action_features"]
        self.assertTrue(np.array_equal(
            [[1, 1], [0, 1]], feature.mask.numpy()))
        self.assertEqual((2, 2, 5), feature.data.shape)


if __name__ == "__main__":
    unittest.main()
