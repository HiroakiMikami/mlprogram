import unittest
from typing import Optional, Tuple
from mlprogram.samplers import Sampler, transform


class MockSampler(Sampler[int, int, str]):
    def create_output(self, input, state: str) -> Optional[Tuple[int, bool]]:
        if len(state) == 0:
            return None
        else:
            return int(state), True


class TestTransform(unittest.TestCase):
    def test_transform(self):
        sampler = transform(MockSampler(), lambda x: x * 2)
        self.assertEqual((4, True), sampler.create_output(None, "2"))
        self.assertEqual(None, sampler.create_output(None, ""))


if __name__ == "__main__":
    unittest.main()
