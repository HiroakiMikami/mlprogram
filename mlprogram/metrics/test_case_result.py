from typing import Dict, Any, Generic, TypeVar, cast
from mlprogram.metrics import Metric, Accuracy
from mlprogram.interpreters import Interpreter, SequentialProgram


Code = TypeVar("Code")
Input = TypeVar("Input")
Result = TypeVar("Result")


class TestCaseResult(Metric[Code], Generic[Code, Input, Result]):
    def __init__(self,
                 interpreter: Interpreter[Code, Input, Result],
                 reference: bool = False,
                 metric: Metric[Result] = Accuracy()):
        self.interpreter = interpreter
        self.reference = reference
        self.metric = metric

    def _eval(self, code: Code, input: Input):
        if self.reference:
            ref = cast(SequentialProgram[Any], code)
            output = self.interpreter.eval_references(ref, input)[
                ref.statements[-1].reference]
        else:
            output = self.interpreter.eval(code, input)
        return output

    def __call__(self, input: Dict[str, Any], value: Code) -> float:
        t_input, output = input["input"]  # get test case

        # calc. metric
        actual = self._eval(value, t_input)
        return self.metric({"ground_truth": output}, actual)
