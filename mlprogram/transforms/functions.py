from typing import Callable, Generic, Optional, TypeVar

from mlprogram import Environment

Code = TypeVar("Code")


class NormalizeGroundTruth(Generic[Code]):
    def __init__(self, normalize: Callable[[Code], Optional[Code]]):
        self.normalize = normalize

    def __call__(self, entry: Environment) -> Environment:
        gt = entry.supervisions["ground_truth"]
        norm_gt = self.normalize(entry.supervisions["ground_truth"])
        if norm_gt is not None:
            gt = norm_gt

        entry.supervisions["ground_truth"] = gt
        return entry
