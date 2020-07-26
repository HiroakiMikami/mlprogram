import numpy as np
import torch
import unittest
from mlprogram.nn.utils.rnn import pad_sequence
from mlprogram.nn.pbe_with_repl import Encoder


class TestEncoder(unittest.TestCase):
    def test_parameters(self):
        encoder = Encoder()
        self.assertEqual(0, len(list(encoder.parameters())))

    def test_shape(self):
        encoder = Encoder()
        output = encoder({
            "input_feature": torch.arange(2).reshape(2, 1),
            "reference_features": pad_sequence([torch.arange(3).reshape(3, 1),
                                                torch.arange(1).reshape(1, 1)
                                                ])
        })
        self.assertEqual((2, 2), output["input_feature"].shape)
        self.assertEqual((3, 2, 1), output["reference_features"].data.shape)
        self.assertTrue(np.array_equal(
            [[1, 1], [1, 0], [1, 0]],
            output["reference_features"].mask.numpy()
        ))

    def test_empty_sequence(self):
        encoder = Encoder()
        output = encoder({
            "input_feature": torch.arange(1).reshape(1, 1),
            "reference_features": pad_sequence([torch.arange(0).reshape(0)])
        })
        self.assertEqual((1, 2), output["input_feature"].shape)
        self.assertEqual((0, 1, 1), output["reference_features"].data.shape)

    def test_mean(self):
        encoder = Encoder()
        mencoder = Encoder("mean")
        output = encoder({
            "input_feature": torch.arange(2).reshape(2, 1),
            "reference_features": pad_sequence([torch.arange(3).reshape(3, 1),
                                                torch.arange(1).reshape(1, 1)
                                                ])
        })
        mean = mencoder({
            "input_feature": torch.arange(2).reshape(2, 1),
            "reference_features": pad_sequence([torch.arange(3).reshape(3, 1),
                                                torch.arange(1).reshape(1, 1)
                                                ])
        })
        self.assertTrue(np.array_equal(
            output["input_feature"][:, 1] / torch.tensor([3.0, 1.0]),
            mean["input_feature"][:, 1],
        ))


if __name__ == "__main__":
    unittest.main()
