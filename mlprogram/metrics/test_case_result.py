from typing import Generic, TypeVar

from mlprogram import Environment
from mlprogram.metrics import Metric, Accuracy
from mlprogram.interpreters import Interpreter


Code = TypeVar("Code")
Input = TypeVar("Input")
Result = TypeVar("Result")
Kind = TypeVar("Kind")


class TestCaseResult(Metric[Code], Generic[Code, Input, Result, Kind]):
    def __init__(self,
                 interpreter: Interpreter[Code, Input, Result, Kind],
                 metric: Metric[Result] = Accuracy()):
        self.interpreter = interpreter
        self.metric = metric

    def _eval(self, code: Code, input: Input):
        output = self.interpreter.eval(code, input)
        return output

    def __call__(self, input: Environment, value: Code) -> float:
        t_input, output = input.inputs["test_case"]

        # calc. metric
        actual = self._eval(value, t_input)
        minput = Environment(supervisions={"ground_truth": output})
        minput.mutable(
            inputs=False, outputs=False,
            states=False, supervisions=False
        )
        return self.metric(minput, actual)
