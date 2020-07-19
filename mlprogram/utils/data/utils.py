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
