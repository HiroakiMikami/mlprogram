from typing import Callable, Generic, Optional, TypeVar

from torch import nn

Code = TypeVar("Code")


class NormalizeGroundTruth(nn.Module, Generic[Code]):
    def __init__(self, normalize: Callable[[Code], Optional[Code]]):
        super().__init__()
        self.normalize = normalize

    def forward(self, ground_truth: Code) -> Code:
        norm_gt = self.normalize(ground_truth)
        if norm_gt is not None:
            ground_truth = norm_gt
        return ground_truth
