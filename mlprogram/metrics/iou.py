import numpy as np
from typing import Dict, Any, Generic, TypeVar, cast, List, Tuple
from mlprogram.metrics import Metric
from mlprogram.interpreters import Interpreter
from mlprogram.utils import Reference

Code = TypeVar("Code")


class Iou(Metric[Code], Generic[Code]):
    def __init__(self, interpreter: Interpreter[Code, np.array],
                 reference: bool = False):
        self.interpreter = interpreter
        self.reference = reference

    def __call__(self, input: Dict[str, Any],
                 actual: Code) -> float:
        test_case = input["test_case"]
        if self.reference:
            actual_ref = cast(List[Tuple[Reference, Any]], actual)
            output = self.interpreter.eval_references(actual_ref)[
                actual_ref[-1][0]]
        else:
            output = self.interpreter.eval(actual)
        iou = 0.0
        intersection = (test_case & output).astype(np.float).sum()
        union = (test_case | output).astype(np.float).sum()
        iou = max(iou, (intersection / union).item())
        return iou
