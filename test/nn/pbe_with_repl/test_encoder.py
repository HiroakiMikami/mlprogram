import numpy as np
import torch
import unittest
from mlprogram.nn.utils.rnn import pad_sequence
from mlprogram.nn.pbe_with_repl import Encoder


class TestEncoder(unittest.TestCase):
    def test_parameters(self):
        encoder = Encoder(torch.nn.ReLU())
        self.assertEqual(0, len(list(encoder.parameters())))

    def test_shape(self):
        encoder = Encoder(torch.nn.Linear(2, 1))
        output = encoder({
            "processed_input": torch.rand(1, 1),
            "input_feature": torch.arange(2).reshape(2, 1),
            "variables": pad_sequence([torch.rand(3, 1),
                                       torch.rand(1, 1),
                                       ])
        })
        self.assertEqual((2, 2), output["input_feature"].shape)
        self.assertEqual((3, 2, 1), output["reference_features"].data.shape)
        self.assertTrue(np.array_equal(
            [[1, 1], [1, 0], [1, 0]],
            output["reference_features"].mask.numpy()
        ))

    def test_empty_sequence(self):
        encoder = Encoder(torch.nn.Linear(2, 1))
        output = encoder({
            "processed_input": torch.rand(1, 1),
            "input_feature": torch.arange(1).reshape(1, 1),
            "variables": pad_sequence([torch.rand(0, 1)])
        })
        self.assertEqual((1, 2), output["input_feature"].shape)
        self.assertEqual((1, 1), output["variable_feature"].shape)
        self.assertEqual((0, 1, 1), output["reference_features"].data.shape)

    def test_mean(self):
        module = torch.nn.Linear(2, 1)
        encoder = Encoder(module)
        mencoder = Encoder(module, "mean")
        input = torch.rand(2, 1)
        variables = pad_sequence([torch.rand(3, 1),
                                  torch.rand(1, 1)
                                  ])
        output = encoder({
            "processed_input": input,
            "input_feature": torch.arange(2).reshape(2, 1),
            "variables": variables
        })
        mean = mencoder({
            "processed_input": input,
            "input_feature": torch.arange(2).reshape(2, 1),
            "variables": variables
        })
        self.assertTrue(np.allclose(
            (output["input_feature"][:, 1] /
             torch.tensor([3.0, 1.0])).detach().numpy(),
            mean["input_feature"][:, 1].detach().numpy(),
        ))

    def test_mask(self):
        encoder = Encoder(torch.nn.Linear(2, 1))
        output = encoder({
            "processed_input": torch.rand(1, 1),
            "input_feature": torch.arange(2).reshape(2, 1),
            "variables": pad_sequence([torch.rand(3, 1),
                                       torch.rand(1, 1)
                                       ])
        })
        self.assertEqual((2, 2), output["input_feature"].shape)
        self.assertEqual((3, 2, 1), output["reference_features"].data.shape)
        padded = output["reference_features"].\
            data[output["reference_features"].mask == 0]
        self.assertTrue(np.array_equal(
            padded.detach().numpy(),
            torch.zeros_like(padded).numpy()
        ))
        self.assertTrue(np.array_equal(
            [[1, 1], [1, 0], [1, 0]],
            output["reference_features"].mask.numpy()
        ))


if __name__ == "__main__":
    unittest.main()
