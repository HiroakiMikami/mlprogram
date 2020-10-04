from typing import TypeVar, Generic, Dict, Any
from mlprogram.metrics.metric import Metric

Value = TypeVar("Value")


class Accuracy(Metric[Value], Generic[Value]):
    def __call__(self, input: Dict[str, Any], value: Value) -> float:
        ground_truth = input["ground_truth"]
        return 1.0 if value == ground_truth else 0.0
