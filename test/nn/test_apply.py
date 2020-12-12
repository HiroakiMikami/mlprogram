import numpy as np
import torch
from torch import nn

from mlprogram import Environment
from mlprogram.nn import Apply


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


class MockModule2(nn.Module):
    def __init__(self, k: int):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(k, dtype=torch.float))

    def forward(self, x):
        assert len(x.shape) == 2
        out = x + self.p
        return out, x


class TestApply(object):
    # TODO: propagate supervision flag
    def test_parameters(self):
        apply = Apply(["x"], "out", MockModule(1))
        assert set(["module.p"]) == \
            dict(apply.named_parameters()).keys()

    def test_simple(self):
        apply = Apply(["x"], "out", MockModule(1))
        output = apply(Environment(
            {"x": torch.arange(3).reshape(-1, 1)}
        ))
        assert np.array_equal(
            [[1], [2], [3]], output["out"].detach().numpy()
        )

    def test_multiple_inputs(self):
        apply = Apply(["x", "y"], "out", MockModule(1))
        output = apply(Environment(
            {"x": torch.arange(3).reshape(-1, 1), "y": 10}
        ))
        assert np.array_equal(
            [[11], [12], [13]], output["out"].detach().numpy()
        )

    def test_multiple_outputs(self):
        apply = Apply(["x"],
                      ["out", "out2"], MockModule2(1))
        output = apply(Environment(
            {"x": torch.arange(3).reshape(-1, 1)}
        ))
        assert np.array_equal(
            [[1], [2], [3]], output["out"].detach().numpy()
        )
        assert np.array_equal(
            [[0], [1], [2]], output["out2"].detach().numpy()
        )

    def test_rename_keys(self):
        apply = Apply([("in", "x")], "out", MockModule(1))
        output = apply(Environment(
            {"in": torch.arange(3).reshape(-1, 1)}
        ))
        assert np.array_equal(
            [[1], [2], [3]], output["out"].detach().numpy()
        )

    def test_constants(self):
        apply = Apply([], "out", MockModule(1),
                      constants={"x": torch.arange(3).reshape(-1, 1)})
        output = apply(Environment())
        assert np.array_equal(
            [[1], [2], [3]], output["out"].detach().numpy()
        )
