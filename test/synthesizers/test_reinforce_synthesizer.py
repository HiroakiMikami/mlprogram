import torch
from typing import cast

from mlprogram.builtins import Environment
from mlprogram.utils.data import Collate, CollateOptions
from mlprogram.synthesizers import REINFORCESynthesizer, Synthesizer, Result


class MockSynthesizer(Synthesizer[Environment, int]):
    def __init__(self):
        self.model = Module()

    def __call__(self, input: Environment, n_required_output=None):
        y = self.model(input)["y"]
        for _ in range(n_required_output):
            out = int(torch.normal(mean=y, std=5).item())
            yield Result(out, torch.abs(out - y).item(), True, 1)


class Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Linear(1, 1, bias=False)

    def forward(self, input: Environment) -> torch.Tensor:
        x = cast(torch.Tensor, input["x"])
        if len(x.shape) == 0:
            x = x.reshape(1, 1)
        y = self.model(x)
        input["y"] = y
        return input


class Loss(torch.nn.Module):
    def forward(self, input: Environment) -> torch.Tensor:
        x = cast(torch.Tensor, input["x"])
        y = cast(torch.Tensor, input["y"])
        return torch.nn.MSELoss()(x, y)


class Reward(torch.nn.Module):
    def forward(self, x: Environment, output: int) -> float:
        return 0.0


def test_REINFORCESynthesizer():
    synthesizer = MockSynthesizer()
    synthesizer.model.model.weight.data[:] = 10.0
    optimizer = torch.optim.SGD(synthesizer.model.parameters(), 0.1)
    synthesizer = REINFORCESynthesizer(
        synthesizer=synthesizer,
        model=synthesizer.model,
        optimizer=optimizer,
        loss_fn=Loss(),
        reward=Reward(),
        collate=Collate(x=CollateOptions(False, 0, 0)).collate,
        n_rollout=1,
        device=torch.device("cpu"),
        baseline_momentum=0.9,
    )
    input = Environment({"x": torch.tensor(1.0)})
    for i, x in enumerate(synthesizer(input)):
        assert i < 100

        if x.output == 1:
            break
