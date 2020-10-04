from mlprogram.interpreters import Interpreter
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


class EvaluateGroundTruth:
    def __init__(self, interpreter: Interpreter, reference: bool = False):
        self.interpreter = interpreter
        self.reference = reference

    def __call__(self, entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        ground_truth = entry["ground_truth"]
        if self.reference:
            results = self.interpreter.eval_references(ground_truth)
            entry["input"] = results[ground_truth.statements[-1].reference]
        else:
            entry["input"] = self.interpreter.eval(ground_truth)
        return entry
