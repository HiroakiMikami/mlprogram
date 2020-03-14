import torch
from dataclasses import dataclass
from typing import List, Any


@dataclass
class Entry:
    input: Any
    ground_truth: Any

    def __hash__(self):
        return hash(self.input) ^ hash(self.ground_truth)

    def __eq__(self, rhs: Any):
        if isinstance(rhs, Entry):
            return self.input == rhs.input and \
                self.ground_truth == rhs.ground_truth
        else:
            return False


Group = List[Entry]


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, elems: List[Any], transform=None):
        self.elems = elems
        self.transform = None

    def __len__(self) -> int:
        return len(self.elems)

    def __getitem__(self, idx):
        item = self.elems[idx]
        if self.transform is not None:
            return self.transform(item)
        return item
