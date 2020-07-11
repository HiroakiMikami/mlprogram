from typing import Callable, List, Dict, Any, Generic, TypeVar
from mlprogram.metrics import Metric


Code = TypeVar("Code")
Value = TypeVar("Value")


class MetricUsingGroundTruth(Metric[Value], Generic[Code, Value]):
    def __init__(self, parse: Callable[[Code], Value],
                 unparse: Callable[[Value], Code]):
        self.parse = parse
        self.unparse = unparse

    def __call__(self, input: Dict[str, Any], value: Value) -> float:
        ground_truth = input["ground_truth"]
        # normalize ground truths
        gts = set()
        for gt in ground_truth:
            gts.add(gt)
            node = self.parse(gt)
            if node is not None:
                code = self.unparse(node)
            else:
                code = None
            if code is not None:
                gts.add(code)

        # normalize value
        code = self.unparse(value)
        if code is None:
            return 0.0

        # calc. metric
        return self.metric(list(gts), code)

    def metric(self, gts: List[Code], value: Code) -> float:
        raise NotImplementedError
