import torch
import torch.nn as nn
from typing import List, Dict, Any
from mlprogram.samplers \
    import Sampler, SamplerWithValueNetwork, SamplerState, \
    DuplicatedSamplerState
from mlprogram.utils.data import CollateOptions, Collate


class MockSampler(Sampler[int, int, str]):
    def initialize(self, input: int) -> str:
        return str(input)

    def batch_k_samples(self, states: List[SamplerState[str]], n: List[int]):
        for state, k in zip(states, n):
            for i in range(k // len(states)):
                x = state.state + str(i)
                yield DuplicatedSamplerState(SamplerState(len(x), x), 1)


class MockValueNetwork(nn.Module):
    def forward(self, state: Dict[str, Any]) -> torch.Tensor:
        return state["x"]


class TestSamplerWithValueNetwork(object):
    def test_rescore(self):
        def transform(state: str) -> torch.Tensor:
            return {"x": torch.tensor([int(state)])}
        collate = Collate(device=torch.device("cpu"),
                          x=CollateOptions(False, 0, 0))
        sampler = SamplerWithValueNetwork(MockSampler(), transform, collate,
                                          MockValueNetwork())
        zero = SamplerState(0, sampler.initialize(0))
        samples = list(sampler.batch_k_samples([zero], [3]))
        assert [DuplicatedSamplerState(SamplerState(0, "00"), 1),
                DuplicatedSamplerState(SamplerState(1, "01"), 1),
                DuplicatedSamplerState(SamplerState(2, "02"), 1)
                ] == samples
