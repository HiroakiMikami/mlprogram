import unittest
from typing import Optional, Tuple
from mlprogram.samplers import Sampler, FilteredSampler


class MockSampler(Sampler[int, int, str]):
    def __init__(self, value):
        self.value = value

    def create_output(self, input, state: str) -> Optional[Tuple[int, bool]]:
        return self.value


class TestFilteredSampler(unittest.TestCase):
    def test_finish_synthesize_if_score_is_high(self):
        sampler = FilteredSampler(
            MockSampler((0, False)),
            lambda x, y: 1.0 if y in x["input"] else y,
            0.9, True)
        output = sampler.create_output({"input": [0]}, None)
        self.assertEqual((0, True), output)

        sampler = FilteredSampler(
            MockSampler(None),
            lambda x, y: 1.0 if y in x["input"] else y,
            0.9, True)
        output = sampler.create_output({"input": [0]}, None)
        self.assertEqual(None, output)

        sampler = FilteredSampler(
            MockSampler((0.1, False)),
            lambda x, y: 1.0 if y in x["input"] else y,
            0.9, True)
        output = sampler.create_output({"input": [0]}, None)
        self.assertEqual((0.1, False), output)

    def test_remove_not_finished_output(self):
        sampler = FilteredSampler(
            MockSampler((0.1, False)),
            lambda x, y: 1.0 if y in x["input"] else y,
            0.9, False)
        output = sampler.create_output({"input": [0]}, None)
        self.assertEqual(None, output)


if __name__ == "__main__":
    unittest.main()
