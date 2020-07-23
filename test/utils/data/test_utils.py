import torch
import unittest
from mlprogram.utils.data import to_map_style_dataset


class MockDataset(torch.utils.data.IterableDataset):
    def __iter__(self):
        class InternalIterator:
            def __init__(self):
                self.n = 0

            def __next__(self) -> int:
                self.n += 1
                return self.n
        return InternalIterator()


class TestToMapStyleDataset(unittest.TestCase):
    def test_happy_path(self):
        dataset = MockDataset()
        elems = to_map_style_dataset(dataset, 10)
        self.assertEqual(10, len(elems))
        self.assertEqual(list(range(1, 11)), list(elems))


if __name__ == "__main__":
    unittest.main()
