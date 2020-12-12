from typing import Callable, Generic, Optional, TypeVar

from mlprogram import Environment

Code = TypeVar("Code")


class NormalizeGroundTruth(Generic[Code]):
    def __init__(self, normalize: Callable[[Code], Optional[Code]]):
        self.normalize = normalize

    def __call__(self, entry: Environment) -> Environment:
        gt = entry["ground_truth"]
        norm_gt = self.normalize(entry["ground_truth"])
        if norm_gt is not None:
            gt = norm_gt

        entry["ground_truth"] = gt
        entry.mark_as_supervision("ground_truth")
        return entry
