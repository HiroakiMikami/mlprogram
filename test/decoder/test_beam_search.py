import unittest
from mlprogram.decoder import BeamSearch, DecoderState, Result
from typing import List, Tuple


class MockBeamSearch(BeamSearch[str, str, Tuple[str, List[str]]]):
    def __init__(self, beam_size: int, max_step_size: int):
        def initialize(input: str) -> Tuple[str, List[str]]:
            return (input, [])

        def create_output(state: Tuple[str, List[str]]):
            x = "".join(state[1])
            if "0" not in x:
                return None
            else:
                return x

        def top_k_samples(states: List[DecoderState[Tuple[str, List[str]]]],
                          k: int):
            for s in states[:k]:
                elems = len(s.state[1])
                gt = s.state[0][elems]
                yield DecoderState(s.score + 0.0,
                                   (s.state[0], s.state[1] + [gt]))
            s = states[0]
            for i in range(k - len(states)):
                x = chr(i + ord('0'))
                yield DecoderState(s.score - i - 1,
                                   (s.state[0], s.state[1] + [x]))

        super().__init__(beam_size, max_step_size, initialize, create_output,
                         top_k_samples)


class TestBeamSearch(unittest.TestCase):
    def test_happy_path(self):
        decoder = MockBeamSearch(3, 100)
        results = list(decoder("x0"))
        self.assertEqual(
            [Result("0", -1.0), Result("x0", 0.0), Result("10", -2.0)],
            results
        )

    def test_abort(self):
        decoder = MockBeamSearch(3, 2)
        results = list(decoder("".join([" "] * 100)))
        self.assertEqual([Result("0", -1.0)], results)


if __name__ == "__main__":
    unittest.main()
