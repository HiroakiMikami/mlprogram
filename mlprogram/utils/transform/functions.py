from typing import Any, Optional, Dict, Callable, TypeVar, Generic

Code = TypeVar("Code")


class NormalizeGroudTruth(Generic[Code]):
    def __init__(self, normalize: Callable[[Code], Optional[Code]]):
        self.normalize = normalize

    def __call__(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        gt = entry["ground_truth"]
        norm_gt = self.normalize(entry["ground_truth"])
        if norm_gt is not None:
            gt = norm_gt

        entry["ground_truth"] = gt
        return entry
