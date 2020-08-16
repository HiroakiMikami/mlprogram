import unittest
from typing import List, Dict, Any
from mlprogram.synthesizers import Synthesizer, Result, SynthesizerWithTimeout
import time


class MockSynthesizer(Synthesizer[Dict[str, Any], int]):
    def __init__(self, values: List[int]):
        self.values = values

    def __call__(self, input: Dict[str, Any], n_required_output=None):
        for i, value in enumerate(self.values):
            time.sleep(2)
            yield Result(value, 1.0 / (i + 1), 1)


class TestFilteredSynthesizer(unittest.TestCase):
    def test_timeout(self):
        synthesizer = SynthesizerWithTimeout(
            MockSynthesizer([0.3, 0.5, 0]),
            1)
        candidates = list(synthesizer({"input": [0]}))
        self.assertEqual(1, len(candidates))
        self.assertEqual(0.3, candidates[0].output)


if __name__ == "__main__":
    unittest.main()
