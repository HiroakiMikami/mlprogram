from typing import List, Optional, Tuple

import numpy as np
import timeout_decorator

from mlprogram.samplers import DuplicatedSamplerState, Sampler, SamplerState
from mlprogram.synthesizers import SMC


class MockSampler(Sampler[str, str, Tuple[str, str]]):
    def __init__(self, rng, finish: bool = False):
        self.rng = rng
        self.finish = finish

    def initialize(self, input: str) -> Tuple[str, str]:
        return (input, "")

    def create_output(self, input, state: Tuple[str, str]):
        x = state[1]
        if "0" not in x:
            return None
        else:
            return x, self.finish

    def batch_k_samples(self, states: List[SamplerState[Tuple[str, str]]],
                        ks: List[int]):
        for state, k in zip(states, ks):
            elems = len(state.state[1])
            if len(state.state[0]) > elems:
                gt: Optional[str] = state.state[0][elems]
            else:
                gt = None
            for _ in range(k // len(states)):
                x = self.rng.choice(['x', 'y', '0', '1'])
                score = 0.0 if gt == x else -1.0
                yield DuplicatedSamplerState(
                    SamplerState(state.score + score,
                                 (state.state[0], state.state[1] + x)),
                    1)


class MockSMC(SMC[str, str, Tuple[str, str], str]):
    def __init__(self, max_step_size: int, max_try_num: int,
                 initial_particle: int, rng: np.random.RandomState,
                 finish: bool = False):
        super().__init__(max_step_size, initial_particle,
                         MockSampler(rng, finish), rng=rng,
                         max_try_num=max_try_num)


class TestSMC(object):
    def test_happy_path(self):
        decoder = MockSMC(3, 10, 10, np.random.RandomState(0))
        results = set([result.output for result in decoder("x0")])
        assert "x0" in results

    def test_when_synthesize_finishes(self):
        decoder = MockSMC(10, 1, 10, np.random.RandomState(0), True)
        results = list([result.output for result in decoder("x0")])
        assert len(results) <= 10

    def test_n_required_output_limit_n_particle(self):
        @timeout_decorator.timeout(1)
        def f():
            decoder = MockSMC(3, 10, int(1e20), np.random.RandomState(0))
            results = set([
                result.output
                for result in decoder("x0", n_required_output=10)])
            assert "x0" in results
        f()
