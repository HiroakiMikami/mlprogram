import numpy as np
import torch
from typing \
    import List, Dict, TypeVar, Generic, Mapping, Any
from dataclasses import dataclass
import time

from mlprogram.synthesizers import Synthesizer
from mlprogram.metrics import Metric


Input = TypeVar("Input")
Code = TypeVar("Code")
GroundTruth = TypeVar("GroundTruth")
DecoderInput = TypeVar("DecoderInput")


@dataclass
class Result(Generic[Input, Code, GroundTruth]):
    query: Input
    data: Dict[str, Any]
    candidates: List[Code]
    metrics: Dict[int, Dict[str, float]]
    generated: bool
    time: float


@dataclass
class EvaluationResult(Generic[Input, Code, GroundTruth]):
    results: List[Result[Input, Code, GroundTruth]]
    metrics: Dict[int, Dict[str, float]]
    generation_rate: float
    generation_time: float


def evaluate(dataset: torch.utils.data.Dataset,
             synthesizer: Synthesizer[Dict[str, Any], Code],
             metrics: Mapping[str, Metric],
             top_n: List[int] = [1, 3],
             ) -> EvaluationResult[Input, Code, GroundTruth]:
    results: List[Result[Input, Code, GroundTruth]] = []
    total = {}
    generated = []
    times = []
    for n in top_n:
        t = {}
        for name in metrics.keys():
            t[name] = 0.0
        total[n] = t
    n_query = 0
    for group in dataset:
        inputs = group["input"]
        for input in inputs:
            n_query += 1
            begin = time.time()
            candidates = list(synthesizer({"input": input}))
            end = time.time()
            generated.append(1.0 if len(candidates) != 0 else 0.0)
            if len(candidates) != 0:
                times.append(end - begin)
            candidates.sort(key=lambda x: -x.score)
            ms = {}
            for n in top_n:
                m: Dict[str, float] = {}
                for name in metrics.keys():
                    m[name] = 0
                for c in candidates[:n]:
                    for name, f in metrics.items():
                        m[name] = max(m[name], f(group, c.output))
                for name in metrics.keys():
                    total[n][name] += \
                        m[name] if m[name] is not None else 0
                ms[n] = m
            results.append(Result(
                input,
                {key: value for key, value in group.items() if key != "input"},
                list(map(lambda x: x.output, candidates)), ms,
                len(candidates) != 0, end - begin))
    total = {n: {name: value / n_query for name, value in metric.items()}
             for n, metric in total.items()}
    return EvaluationResult(results, total, np.mean(generated), np.mean(times))
