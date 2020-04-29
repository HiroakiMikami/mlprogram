import torch
from typing import Callable, Any, List, Dict, Tuple, Mapping
from dataclasses import dataclass


@dataclass
class Result:
    query: str
    ground_truth: List[str]
    metadata: Any
    candidates: List[Any]
    metrics: Dict[int, Dict[str, float]]


@dataclass
class EvaluationResult:
    results: List[Result]
    metrics: Dict[int, Dict[str, float]]


Metric = Callable[[List[Any], Any], float]


def evaluate(dataset: torch.utils.data.Dataset,
             synthesize: Callable[[str], Tuple[Any, List[Any]]],
             metrics: Mapping[str, Metric],
             top_n: List[int] = [1, 3],
             ) -> EvaluationResult:
    results = []
    total = {}
    for n in top_n:
        t = {}
        for name in metrics.keys():
            t[name] = 0.0
        total[n] = t
    for query, ground_truth in dataset:
        metadata, candidates = synthesize(query)
        ms = {}
        for n in top_n:
            m: Dict[str, float] = {}
            for c in candidates[:n]:
                for name, f in metrics.items():
                    if name not in m:
                        m[name] = f(ground_truth, c)
                    else:
                        m[name] = max(m[name], f(ground_truth, c))
            for name in metrics.keys():
                total[n][name] += \
                    m[name] / len(dataset) if m[name] is not None else 0
            ms[n] = m

        results.append(Result(query, ground_truth, metadata, candidates, ms))
    return EvaluationResult(results, total)
