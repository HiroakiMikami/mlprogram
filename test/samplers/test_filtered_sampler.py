from typing import Optional, Tuple
from mlprogram.samplers import Sampler, FilteredSampler


class MockSampler(Sampler[int, int, str]):
    def __init__(self, value):
        self.value = value

    def create_output(self, input, state: str) -> Optional[Tuple[int, bool]]:
        return self.value


class TestFilteredSampler(object):
    def test_finish_synthesize_if_score_is_high(self):
        sampler = FilteredSampler(
            MockSampler((0, False)),
            lambda x, y: 1.0 if y in x["input"] else y,
            0.9)
        output = sampler.create_output({"input": [0]}, None)
        assert (0, True) == output

        sampler = FilteredSampler(
            MockSampler(None),
            lambda x, y: 1.0 if y in x["input"] else y,
            0.9)
        output = sampler.create_output({"input": [0]}, None)
        assert output is None

        sampler = FilteredSampler(
            MockSampler((0.1, False)),
            lambda x, y: 1.0 if y in x["input"] else y,
            0.9)
        output = sampler.create_output({"input": [0]}, None)
        assert (0.1, False) == output
