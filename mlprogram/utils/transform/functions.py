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
        gts = entry["ground_truth"]
        input = []
        for gt in gts:
            if self.reference:
                results = self.interpreter.eval_references(gt)
                input.append(results[gt.statements[-1].reference])
            else:
                input.append(self.interpreter.eval(gt))

        entry["input"] = input
        return entry


class RandomChoice:
    def __init__(self, rng: Optional[np.random.RandomState] = None):
        if rng is None:
            rng = np.random
        self.rng = rng

    def __call__(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        output = {}
        for key, value in entry.items():
            idx = self.rng.randint(0, len(value))
            output[key] = value[idx]
        return output
