import torch
from typing import List, Any, Callable


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, elems: List[Any],
                 transform: Callable[[Any], Any] = None):
        self.elems = elems
        self.transform = None

    def __len__(self) -> int:
        return len(self.elems)

    def __getitem__(self, idx) -> Any:
        item = self.elems[idx]
        if self.transform is not None:
            return self.transform(item)
        return item


def to_map_style_dataset(dataset: torch.utils.data.IterableDataset, n: int) \
        -> ListDataset:
    elems = []
    for i, x in enumerate(dataset):
        if i == n:
            break
        elems.append(x)
    return ListDataset(elems)
