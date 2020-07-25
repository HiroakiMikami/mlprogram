import unittest
from typing import Optional
from mlprogram.samplers import Sampler, transform


class MockSampler(Sampler[int, int, str]):
    def create_output(self, state: str) -> Optional[int]:
        if len(state) == 0:
            return None
        else:
            return int(state)


class TestTransform(unittest.TestCase):
    def test_transform(self):
        sampler = transform(MockSampler(), lambda x: x * 2)
        self.assertEqual(4, sampler.create_output("2"))
        self.assertEqual(None, sampler.create_output(""))


if __name__ == "__main__":
    unittest.main()