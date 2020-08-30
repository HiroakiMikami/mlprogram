from typing import Dict, Any, Generic, TypeVar


Value = TypeVar("Value")


class Metric(Generic[Value]):
    def __call__(self, input: Dict[str, Any], value: Value) -> float:
        raise NotImplementedError
