from typing import Callable, Generic, List, Optional, Tuple, TypeVar

from torch import nn

from mlprogram.builtins.datatypes import Environment
from mlprogram.languages.interpreter import Interpreter
from mlprogram.metrics.accuracy import Accuracy
from mlprogram.metrics.metric import use_environment

Code = TypeVar("Code")
Input = TypeVar("Input")
Result = TypeVar("Result")
Kind = TypeVar("Kind")
Context = TypeVar("Context")


class TestCaseResult(nn.Module, Generic[Code, Input, Result, Kind, Context]):
    def __init__(self,
                 interpreter: Interpreter[Code, Input, Result, Kind, Context],
                 metric: Optional[Callable[[Environment, Result], float]] = None):
        super().__init__()
        self.interpreter = interpreter
        if metric is not None:
            self.metric = metric
        else:
            self.metric = use_environment(
                Accuracy(), in_keys=["actual", "expected"], value_key="actual"
            )

    def forward(self, test_cases: List[Tuple[Input, Kind]], actual: Code) -> float:
        inputs = [input for input, _ in test_cases]
        outputs = [output for _, output in test_cases]

        # calc. metric
        m = 0.0  # TODO reduction function is required
        for actual, expected in zip(self.interpreter.eval(actual, inputs),
                                    outputs):
            m += self.metric(Environment({"expected": expected}), actual)
        return m / len(outputs)
