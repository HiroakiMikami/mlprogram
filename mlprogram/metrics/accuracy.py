from typing import Callable, TypeVar, Generic, Optional
from .metric_using_ground_truth import MetricUsingGroundTruth

Code = TypeVar("Code")
Value = TypeVar("Value")


class Accuracy(MetricUsingGroundTruth[Code, Value], Generic[Code, Value]):
    def __init__(self, parse: Optional[Callable[[Code], Value]],
                 unparse: Optional[Callable[[Value], Code]]):
        super().__init__(parse, unparse)

    def metric(self, gts, value) -> float:
        return 1.0 if value in gts else 0.0
