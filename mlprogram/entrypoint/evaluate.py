import json
import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Generic, List, Mapping, Optional, Tuple, TypeVar

import numpy as np
import torch
from pytorch_pfn_extras.reporting import report
from torch import nn
from tqdm import tqdm

from mlprogram import Environment, logging
from mlprogram.metrics import Metric
from mlprogram.synthesizers import Synthesizer
from mlprogram.utils.data import ListDataset

logger = logging.Logger(__name__)

# This prevents `torch.nn.function.linear`'s from hanging up.
ctx = mp.get_context("spawn")


Code = TypeVar("Code")
GroundTruth = TypeVar("GroundTruth")
DecoderInput = TypeVar("DecoderInput")


@dataclass
class Result(Generic[Code, GroundTruth]):
    sample: Dict[str, Any]
    candidates: List[Code]
    metrics: Dict[int, Dict[str, float]]
    generated: bool
    time: float


@dataclass
class EvaluationResult(Generic[Code, GroundTruth]):
    results: List[Result[Code, GroundTruth]]
    metrics: Dict[int, Dict[str, float]]
    generation_rate: float
    generation_time: float


class EvaluateSample(Generic[Code]):
    def __init__(self, synthesizer: Synthesizer[Environment, Code],
                 metrics: Mapping[str, Metric],
                 top_n: List[int]):
        super().__init__()
        self.synthesizer = synthesizer
        self.metrics = metrics
        self.top_n = top_n

    def __call__(self, elem: Tuple[int, Environment]) \
            -> Result:
        i, sample = elem
        input = Environment(inputs=sample.inputs.to_dict())
        input.mutable(
            inputs=False,
            supervisions=False
        )
        begin = time.time()
        logger.debug(f"Start evaluation of {i}-th sample")
        with logger.block("synthesizer"):
            candidates = list(self.synthesizer(input))
        end = time.time()
        with logger.block("calculate_metrics"):
            candidates.sort(key=lambda x: -x.score)
            ms = {}
            for n in self.top_n:
                m: Dict[str, float] = {}
                for name in self.metrics.keys():
                    m[name] = 0
                for c in candidates[:n]:
                    for name, f in self.metrics.items():
                        m[name] = max(m[name], f(sample, c.output))
                ms[n] = m
        logger.debug(f"Finish evaluation of {i}-th sample")
        return Result(
            sample.to_dict(),
            list(map(lambda x: x.output, candidates)), ms,
            len(candidates) != 0, end - begin)


class EvaluateSynthesizer(Generic[Code, GroundTruth]):
    def __init__(self, dataset: torch.utils.data.Dataset,
                 synthesizer: Synthesizer[Environment, Code],
                 metrics: Mapping[str, Metric], top_n: List[int] = [1, 3],
                 n_process: Optional[int] = None,
                 n_samples: Optional[int] = None):
        super().__init__()
        self.dataset = dataset
        if n_samples is not None:
            self.dataset = ListDataset(self.dataset[:n_samples])
        self.synthesizer = synthesizer
        self.metrics = metrics
        self.top_n = top_n
        self.n_process = n_process

    @logger.function_block("__call__")
    def __call__(self) -> EvaluationResult[Code, GroundTruth]:
        total = {}
        generated = []
        times = []
        for n in self.top_n:
            t = {}
            for name in self.metrics.keys():
                t[name] = 0.0
            total[n] = t
        evaluate_sample: EvaluateSample[Code] = \
            EvaluateSample(self.synthesizer, self.metrics, self.top_n)

        results: List[Result[Code, GroundTruth]] = []
        if self.n_process is None:
            logger.info(f"Evalute with {len(self.dataset)} samples")
            results = [
                evaluate_sample(elem)
                for elem in tqdm(
                    logger.iterable_block("evaluate_sample",
                                          enumerate(self.dataset)))]
        else:
            logger.info(
                f"Evalute with {len(self.dataset)} samples "
                f"using {self.n_process} processes")
            results = []
            with ctx.Pool(processes=self.n_process) as pool:
                with tqdm(total=len(self.dataset)) as _t:
                    for _r in pool.imap_unordered(evaluate_sample,
                                                  enumerate(self.dataset)):
                        _t.update(1)
                        results.append(_r)

        logger.info("Summarize results")
        for result in results:
            generated.append(1.0 if result.generated else 0.0)
            if result.generated:
                times.append(result.time)
            for n in self.top_n:
                m = result.metrics[n]
                for name in self.metrics.keys():
                    total[n][name] += \
                        m[name] if m[name] is not None else 0

        total = {n: {name: value / len(self.dataset)
                     for name, value in metric.items()}
                 for n, metric in total.items()}
        r = EvaluationResult(results, total,
                             np.mean(generated), np.mean(times))
        # report
        for n, metric in total.items():
            for name, value in metric.items():
                report({f"{name}@{n}": value})
        report({"generation_rate": r.generation_rate})
        report({"generation_time": r.generation_time})
        # logging
        logger.info(f"{r.metrics}")
        logger.info(f"generation rate: {r.generation_rate}")
        logger.info(f"generation time: {r.generation_time}")
        return r


def evaluate(input_dir: str, workspace_dir: str, output_dir: str,
             valid_dataset: torch.utils.data.Dataset,
             model: nn.Module,
             synthesizer: Synthesizer,
             metrics: Mapping[str, Metric],
             top_n: List[int] = [1],
             device: torch.device = torch.device("cpu"),
             n_process: Optional[int] = None,
             n_samples: Optional[int] = None) \
        -> None:
    os.makedirs(workspace_dir, exist_ok=True)

    logger.info("Prepare model")
    model.to(device)

    evaluate_synthesizer = EvaluateSynthesizer[Code, GroundTruth](
        valid_dataset, synthesizer, metrics, top_n, n_process, n_samples)

    model_dir = os.path.join(input_dir, "model")
    if len(os.listdir(model_dir)) != 1:
        logger.warning(f"There are multiple models in {model_dir}")
    pathes = []
    for model_path in os.listdir(model_dir):
        model_path = os.path.join(model_dir, os.path.basename(model_path))
        score = \
            torch.load(model_path, map_location=torch.device("cpu"))["score"]
        pathes.append((score, model_path))
    pathes.sort(key=lambda x: -x[0])
    model_path = pathes[0][1]

    logger.info(f"Start evaluation: {model_path}")
    state_dict = \
        torch.load(model_path, map_location=torch.device("cpu"))["model"]
    model.load_state_dict(state_dict)

    result = evaluate_synthesizer()

    logger.info("Save result to output_dir")
    os.makedirs(output_dir, exist_ok=True)
    torch.save(result, os.path.join(output_dir, "result.pt"))
    with open(os.path.join(output_dir, "result_metrics.json"),
              "w") as file:
        json.dump(
            {
                "metrics": result.metrics,
                "generation_rate": result.generation_rate,
                "generation_time": result.generation_time
            },
            file
        )
