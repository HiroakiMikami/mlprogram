from typing import Dict, Any, Generic, TypeVar, List, Tuple, cast
from mlprogram.utils import Reference
from mlprogram.metrics import Metric, Accuracy
from mlprogram.interpreters import Interpreter


Code = TypeVar("Code")
Result = TypeVar("Result")


class TestCaseResult(Metric[Code], Generic[Code]):
    def __init__(self,
                 interpreter: Interpreter[Code, Result],
                 reference: bool = False,
                 metric: Metric[Result] = Accuracy()):
        self.interpreter = interpreter
        self.reference = reference
        self.metric = metric

    def _eval(self, code: Code):
        if self.reference:
            ref = cast(List[Tuple[Reference, Any]], code)
            output = self.interpreter.eval_references(ref)[
                ref[-1][0]]
        else:
            output = self.interpreter.eval(code)
        return output

    def __call__(self, input: Dict[str, Any], value: Code) -> float:
        ground_truth = input["ground_truth"]
        # evaluate ground truth
        outputs = []
        for gt in ground_truth:
            outputs.append(self._eval(gt))

        # calc. metric
        actual = self._eval(value)
        return self.metric({"ground_truth": outputs}, actual)
