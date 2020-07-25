import torch
import unittest
from mlprogram.utils.data import ListDataset, to_map_style_dataset, transform


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


class TestTransform(unittest.TestCase):
    def test_map_style_dataset(self):
        dataset = transform(ListDataset([0]), lambda x: x + 1)
        self.assertEqual(1, len(dataset))
        self.assertEqual(1, dataset[0])

    def test_iterable_dataset(self):
        dataset = transform(MockDataset(), lambda x: x + 1)
        for x in dataset:
            break
        self.assertEqual(2, x)


if __name__ == "__main__":
    unittest.main()
