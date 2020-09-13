from typing import Dict, Any, Generic, TypeVar, Callable


Value = TypeVar("Value")


class Metric(Generic[Value]):
    def __call__(self, input: Dict[str, Any], value: Value) -> float:
        raise NotImplementedError


class TransformedMetric(Metric[Value], Generic[Value]):
    def __init__(self, metric: Metric[Value], f: Callable[[float], float]):
        super().__init__()
        self.metric = metric
        self.f = f

    def __call__(self, input: Dict[str, Any], value: Value) -> float:
        return self.f(self.metric(input, value))


def transform(metric: Metric[Value], transform: Callable[[float], float]):
    return TransformedMetric(metric, transform)
