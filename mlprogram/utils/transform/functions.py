import numpy as np

from mlprogram.interpreters import Interpreter
from typing import Any, Optional, Dict


class EvaluateGroundTruth:
    def __init__(self, interpreter: Interpreter):
        self.interpreter = interpreter

    def __call__(self, entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        gt = entry["ground_truth"]
        if isinstance(gt, list):
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
