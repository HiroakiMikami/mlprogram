from typing import Dict, Any, Generic, TypeVar, cast
from mlprogram.metrics import Metric, Accuracy
from mlprogram.interpreters import Interpreter, SequentialProgram


Code = TypeVar("Code")
Result = TypeVar("Result")


class TestCaseResult(Metric[Code], Generic[Code]):
    def __init__(self,
                 interpreter: Interpreter[Code, Result],
                 reference: bool = False,
                 use_input: bool = False,
                 metric: Metric[Result] = Accuracy()):
        self.interpreter = interpreter
        self.use_input = use_input
        self.reference = reference
        self.metric = metric

    def _eval(self, code: Code):
        if self.reference:
            ref = cast(SequentialProgram[Any], code)
            output = self.interpreter.eval_references(ref)[
                ref.statements[-1].reference]
        else:
            output = self.interpreter.eval(code)
        return output

    def __call__(self, input: Dict[str, Any], value: Code) -> float:
        if self.use_input:
            output = input["input"]
        else:
            ground_truth = input["ground_truth"]
            # evaluate ground truth
            output = self._eval(ground_truth)

        # calc. metric
        actual = self._eval(value)
        return self.metric({"ground_truth": output}, actual)
