import torch
import numpy as np

from mlprogram import Environment
from mlprogram.nn.nl2code import ActionSequenceReader
from mlprogram.nn.utils import rnn


class TestActionSequenceReader(object):
    def test_parameters(self):
        reader = ActionSequenceReader(2, 3, 5, 2, 3)
        assert 3 == len(dict(reader.named_parameters()))

    def test_shape(self):
        """
        num_rules = 2
        num_tokens = 3
        num_node_types = 5
        """
        reader = ActionSequenceReader(2, 3, 5, 2, 3)
        action0 = torch.LongTensor([[0, 0, 0]])
        action1 = torch.LongTensor([[0, 0, 0], [1, 1, 1]])
        action = rnn.pad_sequence([action0, action1])  # (2, 2, 3)
        prev_action0 = torch.LongTensor([[0, 0, 0]])
        prev_action1 = torch.LongTensor([[0, 0, 0], [1, 1, 1]])
        prev_action = rnn.pad_sequence(
            [prev_action0, prev_action1])  # (2, 2, 3)

        data = reader(Environment(states={"actions": action,
                                          "previous_actions": prev_action}))
        feature = data.states["action_features"]
        index = data.states["parent_indexes"]
        assert np.array_equal(
            [[1, 1], [0, 1]], feature.mask.numpy())
        assert (2, 2, 8) == feature.data.shape
        assert np.array_equal(
            [[1, 1], [0, 1]], index.mask.numpy())
        assert (2, 2, 1) == index.data.shape
