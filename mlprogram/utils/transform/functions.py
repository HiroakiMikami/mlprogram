import numpy as np

from mlprogram.interpreters import Interpreter
from typing import Any, Optional, Dict, Callable, TypeVar, Generic

Code = TypeVar("Code")


class NormalizeGroudTruth(Generic[Code]):
    def __init__(self, normalize: Callable[[Code], Optional[Code]]):
        self.normalize = normalize

    def __call__(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        gts = []
        for gt in entry["ground_truth"]:
            norm_gt = self.normalize(gt)
            if norm_gt is None:
                norm_gt = gt
            gts.append(norm_gt)
        entry["ground_truth"] = gts
        return entry


class EvaluateGroundTruth:
    def __init__(self, interpreter: Interpreter, reference: bool = False):
        self.interpreter = interpreter
        self.reference = reference

    def __call__(self, entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        gt = entry["ground_truth"]
        if self.reference:
            results = self.interpreter.eval_references(gt)
            entry["variables"] = results
            entry["test_case"] = results[gt[-1][0]]
        else:
            entry["test_case"] = self.interpreter.eval(gt)
        return entry


class RandomChoice:
    def __init__(self, rng: Optional[np.random.RandomState] = None):
        if rng is None:
            rng = np.random
        self.rng = rng

    def __call__(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        output = {}
        for key, value in entry.items():
            output[key] = self.rng.choice(value, size=()).item()
        return output
