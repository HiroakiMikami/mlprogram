from typing import Generic, TypeVar

from mlprogram import Environment
from mlprogram.languages import Analyzer, Interpreter
from mlprogram.metrics.metric import Metric

Code = TypeVar("Code")
Error = TypeVar("Error")
Diff = TypeVar("Diff")
Input = TypeVar("Input")
Kind = TypeVar("Kind")


class ErrorCorrectRate(Metric[Diff], Generic[Code, Error, Diff, Input, Kind]):
    def __init__(self, analyzer: Analyzer[Code, Error],
                 interpreter: Interpreter[Diff, Input, Code, Kind]):
        self.analyzer = analyzer
        self.interpreter = interpreter

    def __call__(self, input: Environment, value: Diff) -> float:
        original = input["test_cases"][0][0]
        n_orig_error = len(self.analyzer(original))
        fixed = self.interpreter.eval(value, [original])[0]

        n_error = len(self.analyzer(fixed))

        if n_orig_error == 0:
            return 1.0 if n_error == 0 else 0.0

        if n_orig_error < n_error:
            return 0.0
        else:
            return (n_orig_error - n_error) / n_orig_error
