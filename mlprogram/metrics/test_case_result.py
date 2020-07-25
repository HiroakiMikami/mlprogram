from typing import Callable, Dict, Any, Generic, TypeVar
from mlprogram.metrics import Metric, Accuracy
from mlprogram.interpreters import Interpreter


Code = TypeVar("Code")
Value = TypeVar("Value")
Result = TypeVar("Result")


class TestCaseResult(Metric[Value], Generic[Code, Value]):
    def __init__(self, unparse: Callable[[Value], Code],
                 interpreter: Interpreter[Code, Result],
                 metric: Metric[Result] = Accuracy()):
        self.unparse = unparse
        self.interpreter = interpreter
        self.metric = metric

    def __call__(self, input: Dict[str, Any], value: Value) -> float:
        ground_truth = input["ground_truth"]
        # evaluate ground truth
        outputs = set()
        for gt in ground_truth:
            outputs.add(self.interpreter.eval(gt))

        # normalize value
        code = self.unparse(value)
        if code is None:
            return 0.0

        # calc. metric
        actual = self.interpreter.eval(code)
        return self.metric({"ground_truth": outputs}, actual)
