from typing import Optional, Callable, TypeVar, Generic

from mlprogram import Environment

Code = TypeVar("Code")


class NormalizeGroudTruth(Generic[Code]):
    def __init__(self, normalize: Callable[[Code], Optional[Code]]):
        self.normalize = normalize

    def __call__(self, entry: Environment) -> Environment:
        gt = entry.supervisions["ground_truth"]
        norm_gt = self.normalize(entry.supervisions["ground_truth"])
        if norm_gt is not None:
            gt = norm_gt

        entry.supervisions["ground_truth"] = gt
        return entry
