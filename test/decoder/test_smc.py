import unittest
from mlprogram.decoder import SMC, DecoderState
from typing import Tuple, Optional
import numpy as np


class MockSMC(SMC[str, str, Tuple[str, str]]):
    def __init__(self, max_step_size: int, max_retry_num: int,
                 initial_particle: int, rng: np.random.RandomState):
        def initialize(input: str) -> Tuple[str, str]:
            return (input, "")

        def create_output(state: Tuple[str, str]):
            x = state[1]
            if "0" not in x:
                return None
            else:
                return x

        def random_samples(state: DecoderState[Tuple[str, str]],
                           k: int):
            elems = len(state.state[1])
            if len(state.state[0]) < elems:
                gt: Optional[str] = state.state[0][elems]
            else:
                gt = None
            for _ in range(k):
                x = rng.choice(['x', 'y', '0', '1'])
                score = 0.0 if gt == x else -1.0
                yield DecoderState(state.score + score,
                                   (state.state[0], state.state[1] + x))

        super().__init__(max_step_size, max_retry_num, initial_particle,
                         initialize, create_output, random_samples, rng=rng)


class TestBeamSearch(unittest.TestCase):
    def test_happy_path(self):
        decoder = MockSMC(3, 10, 10, np.random.RandomState(0))
        results = set([result.output for result in decoder("x0")])
        self.assertTrue("x0" in results)


if __name__ == "__main__":
    unittest.main()
