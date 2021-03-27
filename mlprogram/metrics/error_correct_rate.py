from typing import Any, Generic, List, Tuple, TypeVar

from torch import nn

from mlprogram.languages.analyzer import Analyzer
from mlprogram.languages.interpreter import Interpreter

Code = TypeVar("Code")
Error = TypeVar("Error")
Diff = TypeVar("Diff")
Kind = TypeVar("Kind")
Context = TypeVar("Context")


class ErrorCorrectRate(nn.Module, Generic[Code, Error, Diff, Kind, Context]):
    def __init__(self, analyzer: Analyzer[Code, Error],
                 interpreter: Interpreter[Diff, Code, Code, Kind, Context]):
        super().__init__()
        self.analyzer = analyzer
        self.interpreter = interpreter

    def forward(self, test_cases: List[Tuple[Code, Any]], actual: Diff) -> float:
        original = test_cases[0][0]
        n_orig_error = len(self.analyzer(original))
        fixed = self.interpreter.eval(actual, [original])[0]

        n_error = len(self.analyzer(fixed))

        if n_orig_error == 0:
            return 1.0 if n_error == 0 else 0.0

        if n_orig_error < n_error:
            return 0.0
        else:
            return (n_orig_error - n_error) / n_orig_error
