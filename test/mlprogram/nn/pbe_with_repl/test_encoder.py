import numpy as np
import torch

from mlprogram.nn.pbe_with_repl import Encoder
from mlprogram.nn.utils.rnn import pad_sequence


class TestEncoder(object):
    def test_parameters(self):
        encoder = Encoder(torch.nn.ReLU())
        assert 0 == len(list(encoder.parameters()))

    def test_shape(self):
        # batch size=2, sequence length=3, num test cases=4
        # channel=1
        encoder = Encoder(torch.nn.Linear(2, 1))
        ref, input = encoder(
            test_case_tensor=torch.rand(2, 4, 1),
            variables_tensor=pad_sequence([torch.rand(3, 4, 1),
                                           torch.rand(1, 4, 1),
                                           ]),
            test_case_feature=torch.arange(2).reshape(2, 1, 1).expand(2, 4, 1),
        )
        assert (2, 2) == input.shape
        assert (3, 2, 1) == ref.data.shape
        assert np.array_equal(
            [[1, 1], [1, 0], [1, 0]],
            ref.mask.numpy()
        )

    def test_empty_sequence(self):
        encoder = Encoder(torch.nn.Linear(2, 1))
        ref, input = encoder(
            test_case_tensor=torch.rand(1, 4, 1),
            variables_tensor=pad_sequence([torch.rand(0, 0, 1)]),
            test_case_feature=torch.arange(1).reshape(1, 1, 1).expand(1, 4, 1)
        )
        assert (1, 2) == input.shape
        assert (0, 1, 1) == ref.data.shape

    def test_mask(self):
        encoder = Encoder(torch.nn.Linear(2, 1))
        ref, input = encoder(
            test_case_tensor=torch.rand(2, 4, 1),
            variables_tensor=pad_sequence([torch.rand(3, 4, 1),
                                           torch.rand(1, 4, 1)
                                           ]),
            test_case_feature=torch.arange(2).reshape(2, 1, 1).expand(2, 4, 1),
        )
        assert (2, 2) == input.shape
        assert (3, 2, 1) == ref.data.shape
        padded = ref.data[ref.mask == 0]
        assert np.array_equal(
            padded.detach().numpy(),
            torch.zeros_like(padded).numpy()
        )
        assert np.array_equal(
            [[1, 1], [1, 0], [1, 0]],
            ref.mask.numpy()
        )
