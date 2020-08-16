import unittest
from mlprogram.samplers import Sampler, SamplerState, DuplicatedSamplerState
from mlprogram.synthesizers import SMC
from typing import Tuple, Optional, List
import numpy as np
import timeout_decorator


class MockSampler(Sampler[str, str, Tuple[str, str]]):
    def __init__(self, rng):
        self.rng = rng

    def initialize(self, input: str) -> Tuple[str, str]:
        return (input, "")

    def create_output(self, state: Tuple[str, str]):
        x = state[1]
        if "0" not in x:
            return None
        else:
            return x

    def k_samples(self, states: List[SamplerState[Tuple[str, str]]],
                  k: int):
        for state in states:
            elems = len(state.state[1])
            if len(state.state[0]) < elems:
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
                 initial_particle: int, rng: np.random.RandomState):
        super().__init__(max_step_size, initial_particle,
                         MockSampler(rng), rng=rng, max_try_num=max_try_num)


class TestBeamSearch(unittest.TestCase):
    def test_happy_path(self):
        decoder = MockSMC(3, 10, 10, np.random.RandomState(0))
        results = set([result.output for result in decoder("x0")])
        self.assertTrue("x0" in results)

    def test_n_required_output_limit_n_particle(self):
        @timeout_decorator.timeout(1)
        def f():
            decoder = MockSMC(3, 10, int(1e20), np.random.RandomState(0))
            results = set([
                result.output
                for result in decoder("x0", n_required_output=10)])
            self.assertTrue("x0" in results)
        f()


if __name__ == "__main__":
    unittest.main()
