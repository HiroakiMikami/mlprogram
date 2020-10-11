import numpy as np
import torch
from torch import nn
from mlprogram import Environment
from mlprogram.nn import Apply
from mlprogram.nn.utils.rnn import pad_sequence


class MockModule(nn.Module):
    def __init__(self, k: int):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(k, dtype=torch.float))

    def forward(self, x, y=None):
        assert len(x.shape) == 2
        out = x + self.p
        if y is not None:
            out = out + y
        return out


class TestApply(object):
    def test_parameters(self):
        apply = Apply(["input@x"], "output@out", MockModule(1))
        assert set(["module.p"]) == \
            dict(apply.named_parameters()).keys()

    def test_simple(self):
        apply = Apply(["input@x"], "output@out", MockModule(1))
        output = apply(Environment(
            inputs={"x": torch.arange(3).reshape(-1, 1)}
        ))
        assert np.array_equal(
            [[1], [2], [3]], output["output@out"].detach().numpy()
        )

    def test_sequence(self):
        apply = Apply(["input@x"], "output@out", MockModule(1),
                      value_type="list")

        output = apply(Environment(inputs={
            "x": [torch.arange(2).reshape(-1, 1),
                  torch.arange(1).reshape(-1, 1) * 10,
                  torch.arange(3).reshape(-1, 1) * 100]}))
        assert np.array_equal(
            [[1], [2]], output["output@out"][0].detach().numpy()
        )
        assert np.array_equal(
            [[1]], output["output@out"][1].detach().numpy()
        )
        assert np.array_equal(
            [[1], [101], [201]], output["output@out"][2].detach().numpy()
        )

    def test_empty_sequence(self):
        apply = Apply(["input@x"], "output@out", MockModule(1),
                      value_type="list")

        output = apply(Environment(inputs={"x": []}))
        assert [] == output["output@out"]
        output = apply(Environment(inputs={"x": [torch.zeros((0,))]}))
        assert [[]] == output["output@out"]

    def test_padded_sequence(self):
        apply = Apply(["input@x"], "output@out", MockModule(1),
                      value_type="padded_tensor")

        padded = pad_sequence([torch.arange(2).reshape(-1, 1),
                               torch.arange(1).reshape(-1, 1) * 10,
                               torch.arange(3).reshape(-1, 1) * 100],
                              padding_value=-1)
        output = apply(Environment(inputs={"x": padded}))
        assert np.array_equal(
            [
                [[1], [1], [1]],
                [[2], [0], [101]],
                [[0], [0], [201]]
            ],
            output["output@out"].data.detach().numpy()
        )
        assert np.array_equal(
            [
                [1, 1, 1],
                [1, 0, 1],
                [0, 0, 1]
            ],
            output["output@out"].mask.detach().numpy()
        )

    def test_multiple_inputs(self):
        apply = Apply(["input@x", "input@y"], "output@out", MockModule(1))
        output = apply(Environment(
            inputs={"x": torch.arange(3).reshape(-1, 1), "y": 10}
        ))
        assert np.array_equal(
            [[11], [12], [13]], output["output@out"].detach().numpy()
        )

    def test_rename_keys(self):
        apply = Apply([("input@in", "x")], "output@out", MockModule(1))
        output = apply(Environment(
            inputs={"in": torch.arange(3).reshape(-1, 1)}
        ))
        assert np.array_equal(
            [[1], [2], [3]], output["output@out"].detach().numpy()
        )

    def test_constants(self):
        apply = Apply([], "output@out", MockModule(1),
                      constants={"x": torch.arange(3).reshape(-1, 1)})
        output = apply(Environment())
        assert np.array_equal(
            [[1], [2], [3]], output["output@out"].detach().numpy()
        )
