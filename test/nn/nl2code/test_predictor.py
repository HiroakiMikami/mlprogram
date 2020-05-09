import unittest
import torch
import numpy as np

from mlprogram.nn.nl2code import Predictor, ActionSequenceReader
from mlprogram.nn.utils import rnn


class TestPredictor(unittest.TestCase):
    def test_parameters(self):
        reader = ActionSequenceReader(1, 1, 1, 2, 3)
        predictor = Predictor(reader, 1, 2, 3, 5)
        self.assertEqual(17, len(dict(predictor.named_parameters())))

    def test_shape(self):
        reader = ActionSequenceReader(1, 1, 1, 1, 1)
        predictor = Predictor(reader, 1, 2, 3, 5)
        feature0 = torch.rand(2, 3)
        feature1 = torch.rand(1, 3)
        feature = rnn.pad_sequence([feature0, feature1])
        context0 = torch.rand(2, 2)
        context1 = torch.rand(1, 2)
        context = rnn.pad_sequence([context0, context1])
        query0 = torch.rand(3, 2)
        query1 = torch.rand(1, 2)
        query = rnn.pad_sequence([query0, query1])

        rule_pred, token_pred, copy_pred = predictor(
            query, (feature, context))
        self.assertTrue(np.array_equal(
            [[1, 1], [1, 0]], rule_pred.mask.numpy()))
        self.assertEqual((2, 2, 1), rule_pred.data.shape)
        self.assertTrue(np.allclose([[1, 1], [1, 1]], np.sum(
            rule_pred.data.detach().numpy(), axis=2)))
        self.assertEqual((2, 2, 1), token_pred.data.shape)
        self.assertTrue(np.array_equal(
            [[1, 1], [1, 0]], token_pred.mask.numpy()))
        self.assertEqual((2, 2, 3), copy_pred.data.shape)
        self.assertTrue(np.array_equal(
            [[1, 1], [1, 0]], copy_pred.mask.numpy()))
        self.assertTrue(np.allclose([[1, 1], [1, 1]],
                                    np.sum(token_pred.data.detach().numpy(),
                                           axis=2) +
                                    np.sum(copy_pred.data.detach().numpy(),
                                           axis=2)))


if __name__ == "__main__":
    unittest.main()
