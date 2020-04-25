import torch
from dataclasses import dataclass
from typing import List, TypeVar, Generic, Union, Optional, Callable


Input = TypeVar("Input")
Code = TypeVar("Code")
Elem = TypeVar("Elem")
Transformed = TypeVar("Transformed")


@dataclass
class Entry(Generic[Input, Code]):
    input: Input
    ground_truth: Code

    def __hash__(self) -> int:
        return hash(self.input) ^ hash(self.ground_truth)

    def __eq__(self, rhs: object) -> bool:
        if isinstance(rhs, Entry):
            return self.input == rhs.input and \
                self.ground_truth == rhs.ground_truth
        else:
            return False


Group = List[Entry]


class ListDataset(torch.utils.data.Dataset, Generic[Elem, Transformed]):
    def __init__(self, elems: List[Elem],
                 transform: Optional[Callable[[Elem], Transformed]] = None):
        self.elems = elems
        self.transform = None

    def __len__(self) -> int:
        return len(self.elems)

    def __getitem__(self, idx) -> Union[Elem, Transformed]:
        item = self.elems[idx]
        if self.transform is not None:
            return self.transform(item)
        return item
