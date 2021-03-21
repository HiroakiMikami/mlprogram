import torch

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


class TestToMapStyleDataset(object):
    def test_happy_path(self):
        dataset = MockDataset()
        elems = to_map_style_dataset(dataset, 10)
        assert 10 == len(elems)
        assert list(range(1, 11)) == list(elems)


class TestTransform(object):
    def test_map_style_dataset(self):
        dataset = transform(ListDataset([0]), lambda x: x + 1)
        assert 1 == len(dataset)
        assert 1 == dataset[0]
        assert [1] == dataset[:1]

    def test_iterable_dataset(self):
        dataset = transform(MockDataset(), lambda x: x + 1)
        for x in dataset:
            break
        assert 2 == x
