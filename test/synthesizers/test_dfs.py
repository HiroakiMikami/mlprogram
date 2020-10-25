from mlprogram.samplers import Sampler, SamplerState, DuplicatedSamplerState
from mlprogram.synthesizers import DFS, Result
from typing import List, Tuple


class MockSampler(Sampler[str, str, Tuple[str, List[int]]]):
    def __init__(self, finish: bool = True):
        self.finish = finish

    def initialize(self, input: str) -> Tuple[str, List[int]]:
        return (input, [])

    def create_output(self, input, state: Tuple[str, List[int]]):
        x = state[1]
        if 0 not in x:
            return None
        else:
            return "".join(map(str, x)), self.finish

    def all_samples(self, states: List[SamplerState[Tuple[str, List[int]]]],
                    sorted: bool = True):
        for s in states:
            elems = len(s.state[1])
            for i in range(3 - elems):
                yield DuplicatedSamplerState(
                    SamplerState(s.score + (3 - i),
                                 (s.state[0], s.state[1] + [i])),
                    1)


class TestDFS(object):
    def test_happy_path(self):
        decoder = DFS(MockSampler())
        results = list(decoder("x0"))
        assert results == [
            Result("0", 3.0, True, 1),
            Result("10", 5.0, True, 1),
            Result("110", 7.0, True, 1),
            Result("20", 4.0, True, 1),
            Result("210", 6.0, True, 1)]
