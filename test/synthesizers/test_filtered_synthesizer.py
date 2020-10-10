from typing import List, Dict, Any
from mlprogram.synthesizers import Synthesizer, Result, FilteredSynthesizer


class MockSynthesizer(Synthesizer[Dict[str, Any], int]):
    def __init__(self, values: List[int]):
        self.values = values

    def __call__(self, input: Dict[str, Any], n_required_output=None):
        for i, value in enumerate(self.values):
            yield Result(value, 1.0 / (i + 1), True, 1)


class TestFilteredSynthesizer(object):
    def test_finish_synthesize_if_not_filtered_output_found(self):
        synthesizer = FilteredSynthesizer(
            MockSynthesizer([0.3, 0.5, 0]),
            lambda x, y: 1.0 if y in x["input"] else y,
            0.9)
        candidates = list(synthesizer({"input": [0]}))
        assert 1 == len(candidates)
        assert 0 == candidates[0].output
