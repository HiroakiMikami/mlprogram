import torch
from typing \
    import Callable, List, Dict, TypeVar, Generic, Mapping, Any
from dataclasses import dataclass

from mlprogram.synthesizers import Synthesizer


Code = TypeVar("Code")
GroundTruth = TypeVar("GroundTruth")
DecoderInput = TypeVar("DecoderInput")


@dataclass
class Result(Generic[Code, GroundTruth]):
    query: str
    ground_truth: List[GroundTruth]
    candidates: List[Code]
    metrics: Dict[int, Dict[str, float]]


@dataclass
class EvaluationResult(Generic[Code, GroundTruth]):
    results: List[Result[Code, GroundTruth]]
    metrics: Dict[int, Dict[str, float]]


Metric = Callable[[List[GroundTruth], Code], float]


def evaluate(dataset: torch.utils.data.Dataset,
             synthesizer: Synthesizer[Dict[str, Any], Code],
             metrics: Mapping[str, Metric],
             top_n: List[int] = [1, 3],
             ) -> EvaluationResult[Code, GroundTruth]:
    results: List[Result[Code, GroundTruth]] = []
    total = {}
    for n in top_n:
        t = {}
        for name in metrics.keys():
            t[name] = 0.0
        total[n] = t
    n_query = 0
    for group in dataset:
        queries = group["input"]
        gts: List[GroundTruth] = group["ground_truth"]
        for query in queries:
            n_query += 1
            candidates = list(synthesizer({"input": query}))
            candidates.sort(key=lambda x: -x.score)
            ms = {}
            for n in top_n:
                m: Dict[str, float] = {}
                for name in metrics.keys():
                    m[name] = 0
                for c in candidates[:n]:
                    for name, f in metrics.items():
                        m[name] = max(m[name], f(gts, c.output))
                for name in metrics.keys():
                    total[n][name] += \
                        m[name] if m[name] is not None else 0
                ms[n] = m
            results.append(Result(query, gts,
                                  list(map(lambda x: x.output, candidates)),
                                  ms))
    total = {n: {name: value / n_query for name, value in metric.items()}
             for n, metric in total.items()}
    return EvaluationResult(results, total)
