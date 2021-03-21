import time
from typing import Any, Dict, List

from mlprogram.synthesizers import Result, Synthesizer, SynthesizerWithTimeout


class MockSynthesizer(Synthesizer[Dict[str, Any], int]):
    def __init__(self, values: List[int]):
        self.values = values

    def __call__(self, input: Dict[str, Any], n_required_output=None):
        for i, value in enumerate(self.values):
            time.sleep(2)
            yield Result(value, 1.0 / (i + 1), True, 1)


class TestFilteredSynthesizer(object):
    def test_timeout(self):
        synthesizer = SynthesizerWithTimeout(
            MockSynthesizer([0.3, 0.5, 0]),
            1)
        candidates = list(synthesizer({"input": [0]}))
        assert 1 == len(candidates)
        assert 0.3 == candidates[0].output
