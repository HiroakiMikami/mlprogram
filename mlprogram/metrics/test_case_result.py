from typing import Generic, TypeVar

from mlprogram import Environment
from mlprogram.languages import Interpreter
from mlprogram.metrics import Accuracy, Metric

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

    def __call__(self, input: Environment, value: Code) -> float:
        test_cases = input["test_cases"]
        inputs = [input for input, _ in test_cases]
        outputs = [output for _, output in test_cases]

        # calc. metric
        m = 0.0  # TODO reduction function is required
        for actual, expected in zip(self.interpreter.eval(value, inputs),
                                    outputs):
            minput = Environment({"ground_truth": expected}, set(["ground_truth"]))
            m += self.metric(minput, actual)
        return m / len(outputs)
