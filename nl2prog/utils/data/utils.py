import torch
from dataclasses import dataclass
from typing import List, Any, Callable


@dataclass
class Entry:
    input: Any
    ground_truth: Any

    def __hash__(self) -> int:
        return hash(self.input) ^ hash(self.ground_truth)

    def __eq__(self, rhs: Any) -> bool:
        if isinstance(rhs, Entry):
            return self.input == rhs.input and \
                self.ground_truth == rhs.ground_truth
        else:
            return False


Group = List[Entry]


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
