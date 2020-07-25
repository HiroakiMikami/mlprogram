from typing import TypeVar, Generic, Dict, Any
from .metric import Metric

Value = TypeVar("Value")


class Accuracy(Metric[Value], Generic[Value]):
    def __call__(self, input: Dict[str, Any], value: Value) -> float:
        gts = input["ground_truth"]
        return 1.0 if value in gts else 0.0
