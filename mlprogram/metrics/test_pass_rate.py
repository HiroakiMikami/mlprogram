from typing import Callable, Dict, Any, Generic, TypeVar
from mlprogram.metrics import Metric
from mlprogram.interpreters import Interpreter


Code = TypeVar("Code")
Value = TypeVar("Value")
Result = TypeVar("Result")


class TestPassRate(Metric[Value], Generic[Code, Value]):
    def __init__(self, unparse: Callable[[Value], Code],
                 interpreter: Interpreter[Code, Result]):
        self.unparse = unparse
        self.interpreter = interpreter

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
        return 1.0 if actual in outputs else 0.0
