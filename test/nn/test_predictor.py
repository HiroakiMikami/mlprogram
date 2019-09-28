import unittest
import torch
import numpy as np

from nl2code.nn import Predictor
from nl2code.nn.utils import rnn


class TestPredictor(unittest.TestCase):
    def test_parameters(self):
        predictor = Predictor(2, 3, 5, 7, 2, 3, 2, 3, 5, 0.0)
        self.assertEqual(25, len(dict(predictor.named_parameters())))

    def test_shape(self):
        """
        num_rules = 2
        num_tokens = 3
        num_node_types = 5
        max_query_length = 7
        query_size = 2
        """
        predictor = Predictor(2, 3, 5, 7, 2, 3, 2, 3, 5, 0.0)
        query0 = torch.rand(3, 2)
        query1 = torch.rand(1, 2)
        query = rnn.pad_sequence([query0, query1])  # (3, 2, 2)
        action0 = torch.LongTensor([[0, 0, 0]])
        action1 = torch.LongTensor([[0, 0, 0], [1, 1, 1]])
        action = rnn.pad_sequence([action0, action1])  # (2, 2, 3)
        prev_action0 = torch.LongTensor([[0, 0, 0]])
        prev_action1 = torch.LongTensor([[0, 0, 0], [1, 1, 1]])
        prev_action = rnn.pad_sequence(
            [prev_action0, prev_action1])  # (2, 2, 3)
        history = torch.rand(3, 2, 3)
        h_0 = torch.rand(2, 3)
        c_0 = torch.rand(2, 3)

        rule_pred, token_pred, copy_pred, new_history, (h_n, c_n) = predictor(
            query, action, prev_action, history, (h_0, c_0))
        self.assertTrue(np.array_equal(
            [[1, 1], [0, 1]], rule_pred.mask.numpy()))
        self.assertEqual((2, 2, 2), rule_pred.data.shape)
        self.assertTrue(np.allclose([[1, 1], [1, 1]], np.sum(
            rule_pred.data.detach().numpy(), axis=2)))
        self.assertEqual((2, 2, 3), token_pred.data.shape)
        self.assertTrue(np.array_equal(
            [[1, 1], [0, 1]], token_pred.mask.numpy()))
        self.assertEqual((2, 2, 7), copy_pred.data.shape)
        self.assertTrue(np.array_equal(
            [[1, 1], [0, 1]], copy_pred.mask.numpy()))
        self.assertTrue(np.allclose(
            0, copy_pred.data.detach().numpy()[:, :, 3:]))
        self.assertTrue(np.allclose([[1, 1], [1, 1]],
                                    np.sum(token_pred.data.detach().numpy(),
                                           axis=2) +
                                    np.sum(copy_pred.data.detach().numpy(),
                                           axis=2)))
        self.assertEqual((5, 2, 3), new_history.shape)
        self.assertEqual((2, 3), h_n.shape)
        self.assertEqual((2, 3), c_n.shape)


if __name__ == "__main__":
    unittest.main()
