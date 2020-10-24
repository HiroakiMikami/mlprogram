import torch
from typing import List, Callable, TypeVar, Generic

V = TypeVar("V")
V0 = TypeVar("V0")
V1 = TypeVar("V1")


class ListDataset(torch.utils.data.Dataset, Generic[V]):
    def __init__(self, elems: List[V]):
        self.elems = elems

    def __len__(self) -> int:
        return len(self.elems)

    def __getitem__(self, idx) -> V:
        return self.elems[idx]


class TransformedDataset(torch.utils.data.Dataset, Generic[V0, V1]):
    def __init__(self, dataset: torch.utils.data.Dataset,
                 transform: Callable[[V0], V1]):
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> V1:
        if isinstance(index, int):
            return self.transform(self.dataset[index])
        else:
            return [self.transform(entry) for entry in self.dataset[index]]


class TransformedIterableDataset(torch.utils.data.IterableDataset,
                                 Generic[V0, V1]):
    def __init__(self, dataset: torch.utils.data.IterableDataset,
                 transform: Callable[[V0], V1]):
        self.dataset = dataset
        self.transform = transform

    def __iter__(self):
        class InternalIterator:
            def __init__(self, parent):
                self.transform = parent.transform
                self.iter = iter(parent.dataset)

            def __next__(self) -> V1:
                return self.transform(next(self.iter))
        return InternalIterator(self)


def to_map_style_dataset(dataset: torch.utils.data.IterableDataset, n: int) \
        -> ListDataset:
    elems = []
    for i, x in enumerate(dataset):
        if i == n:
            break
        elems.append(x)
    return ListDataset(elems)


def transform(dataset: torch.utils.data.Dataset,
              transform: Callable[[V0], V1]) -> torch.utils.data.Dataset:
    if hasattr(dataset, "__len__"):
        return TransformedDataset(dataset, transform)
    else:
        return TransformedIterableDataset(dataset, transform)
