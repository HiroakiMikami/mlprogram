import numpy as np
import torch

from mlprogram import Environment
from mlprogram.nn.utils.rnn import pad_sequence
from mlprogram.nn.pbe_with_repl import Encoder


class TestEncoder(object):
    def test_parameters(self):
        encoder = Encoder(torch.nn.ReLU())
        assert 0 == len(list(encoder.parameters()))

    def test_shape(self):
        encoder = Encoder(torch.nn.Linear(2, 1))
        output = encoder(Environment(
            states={
                "test_case_tensor": torch.rand(1, 1),
                "variables_tensor": pad_sequence([torch.rand(3, 1),
                                                  torch.rand(1, 1),
                                                  ]),
                "input_feature": torch.arange(2).reshape(2, 1),
            }))
        assert (2, 2) == output.states["input_feature"].shape
        assert (3, 2, 1) == output.states["reference_features"].data.shape
        assert np.array_equal(
            [[1, 1], [1, 0], [1, 0]],
            output.states["reference_features"].mask.numpy()
        )

    def test_empty_sequence(self):
        encoder = Encoder(torch.nn.Linear(2, 1))
        output = encoder(Environment(
            states={
                "test_case_tensor": torch.rand(1, 1),
                "variables_tensor": pad_sequence([torch.rand(0, 1)]),
                "input_feature": torch.arange(1).reshape(1, 1)
            }
        ))
        assert (1, 2) == output.states["input_feature"].shape
        assert (1, 1) == output.states["variable_feature"].shape
        assert \
            (0, 1, 1) == output.states["reference_features"].data.shape

    def test_mean(self):
        module = torch.nn.Linear(2, 1)
        encoder = Encoder(module)
        mencoder = Encoder(module, "mean")
        input = torch.rand(2, 1)
        variables = pad_sequence([torch.rand(3, 1),
                                  torch.rand(1, 1)
                                  ])
        output = encoder(Environment(
            states={
                "test_case_tensor": input,
                "variables_tensor": variables,
                "input_feature": torch.arange(2).reshape(2, 1),
            }))
        mean = mencoder(Environment(
            states={
                "test_case_tensor": input,
                "variables_tensor": variables,
                "input_feature": torch.arange(2).reshape(2, 1),
            }))
        assert np.allclose(
            (output.states["input_feature"][:, 1] /
             torch.tensor([3.0, 1.0])).detach().numpy(),
            mean.states["input_feature"][:, 1].detach().numpy(),
        )

    def test_mask(self):
        encoder = Encoder(torch.nn.Linear(2, 1))
        output = encoder(Environment(
            states={
                "test_case_tensor": torch.rand(1, 1),
                "variables_tensor": pad_sequence([torch.rand(3, 1),
                                                  torch.rand(1, 1)
                                                  ]),
                "input_feature": torch.arange(2).reshape(2, 1),
            }))
        assert (2, 2) == output.states["input_feature"].shape
        assert (3, 2, 1) == output.states["reference_features"].data.shape
        padded = output.states["reference_features"]. \
            data[output.states["reference_features"].mask == 0]
        assert np.array_equal(
            padded.detach().numpy(),
            torch.zeros_like(padded).numpy()
        )
        assert np.array_equal(
            [[1, 1], [1, 0], [1, 0]],
            output.states["reference_features"].mask.numpy()
        )
