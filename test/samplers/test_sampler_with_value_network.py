import torch
import torch.nn as nn
import unittest
from typing import List
from mlprogram.samplers import Sampler, SamplerWithValueNetwork, SamplerState


class MockSampler(Sampler[int, int, str]):
    def initialize(self, input: int) -> str:
        return str(input)

    def k_samples(self, states: List[SamplerState[str]], n: int):
        for state in states:
            for i in range(n // len(states)):
                x = state.state + str(i)
                yield SamplerState(len(x), x)


class MockValueNetwork(nn.Module):
    def forward(self, state: str) -> torch.Tensor:
        return torch.tensor(int(state))


class TestSamplerWithValueNetwork(unittest.TestCase):
    def test_rescore(self):
        sampler = SamplerWithValueNetwork(MockSampler(), MockValueNetwork())
        zero = SamplerState(0, sampler.initialize(0))
        samples = list(sampler.k_samples([zero], 3))
        self.assertEqual(
            [SamplerState(0, "00"), SamplerState(1, "01"),
             SamplerState(2, "02")],
            samples
        )


if __name__ == "__main__":
    unittest.main()
