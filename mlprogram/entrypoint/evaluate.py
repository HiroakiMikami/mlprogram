import time
from dataclasses import dataclass
import multiprocessing as mp
from typing \
    import List, Dict, TypeVar, Generic, Mapping, Any, Optional, Tuple
import numpy as np
import torch
from torch import nn
import os
import shutil
from pytorch_pfn_extras.reporting import report
from mlprogram.metrics import Metric
from mlprogram.synthesizers import Synthesizer
from mlprogram.utils.data import ListDataset
from mlprogram.utils import logging


logger = logging.Logger(__name__)

# This prevents `torch.nn.function.linear`'s from hanging up.
ctx = mp.get_context("spawn")


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


class EvaluateSample(Generic[Input, Code]):
    def __init__(self, synthesizer: Synthesizer[Dict[str, Any], Code],
                 metrics: Mapping[str, Metric],
                 top_n: List[int]):
        super().__init__()
        self.synthesizer = synthesizer
        self.metrics = metrics
        self.top_n = top_n

    def __call__(self, elem: Tuple[Input, Dict[str, Any]]) -> Result:
        input, group = elem
        begin = time.time()
        with logger.block("synthesizer"):
            candidates = list(self.synthesizer({"input": input}))
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
                        m[name] = max(m[name], f(group, c.output))
                ms[n] = m
        return Result(
            input,
            {key: value for key, value in group.items() if key != "input"},
            list(map(lambda x: x.output, candidates)), ms,
            len(candidates) != 0, end - begin)


class EvaluateSynthesizer(Generic[Input, Code, GroundTruth]):
    def __init__(self, dataset: torch.utils.data.Dataset,
                 synthesizer: Synthesizer[Dict[str, Any], Code],
                 metrics: Mapping[str, Metric], top_n: List[int] = [1, 3],
                 n_process: Optional[int] = None):
        super().__init__()
        self.dataset = dataset
        self.synthesizer = synthesizer
        self.metrics = metrics
        self.top_n = top_n
        self.n_process = n_process

    @logger.function_block("__call__")
    def __call__(self) -> EvaluationResult[Input, Code, GroundTruth]:
        results: List[Result[Input, Code, GroundTruth]] = []
        total = {}
        generated = []
        times = []
        for n in self.top_n:
            t = {}
            for name in self.metrics.keys():
                t[name] = 0.0
            total[n] = t
        evaluate_sample = \
            EvaluateSample[Dict[str, Any], Code](
                self.synthesizer, self.metrics, self.top_n)
        inputs = []
        for group in self.dataset:
            for input in group["input"]:
                inputs.append((input, group))

        if self.n_process is None:
            logger.info(f"Evalute with {len(inputs)} samples")
            results = [evaluate_sample(elem)
                       for elem in logger.iterable_block("evaluate_sample",
                                                         inputs)]
        else:
            logger.info(
                f"Evalute with {len(inputs)} samples "
                f"using {self.n_process} processes")
            with ctx.Pool(processes=self.n_process) as pool:
                list(pool.map(evaluate_sample, inputs))
            results = [evaluate_sample(elem)
                       for elem in logger.iterable_block("evaluate_sample",
                                                         inputs)]

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

        total = {n: {name: value / len(inputs)
                     for name, value in metric.items()}
                 for n, metric in total.items()}
        r = EvaluationResult(results, total,
                             np.mean(generated), np.mean(times))
        for n, metric in total.items():
            for name, value in metric.items():
                report({f"{name}@{n}": value})
        report({"generation_rate": r.generation_rate})
        report({"generation_time": r.generation_time})
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

    if n_samples is not None:
        valid_dataset = ListDataset(valid_dataset[:n_samples])

    logger.info("Prepare model")
    model.to(device)

    evaluate_synthesizer = EvaluateSynthesizer[Input, Code, GroundTruth](
        valid_dataset, synthesizer, metrics, top_n, n_process)

    # Move parameters to shared memory
    if n_process is not None:
        for k, v in model.state_dict().items():
            v.share_memory_()

    model_dir = os.path.join(input_dir, "model")
    result_path = os.path.join(workspace_dir, "result.pt")
    if len(os.listdir(model_dir)) != 1:
        logger.warning(f"There are multiple models in {model_dir}")
    pathes = []
    for model_path in os.listdir(model_dir):
        model_path = os.path.join(model_dir, model_path)
        score = \
            torch.load(model_path, map_location=torch.device("cpu"))["score"]
        pathes.append((score, model_path))
    pathes.sort(key=lambda x: -x[0])
    model_path = pathes[0][1]

    logger.info(f"Start evaluation (valid dataset): {model_path}")
    model_path = os.path.join(model_dir, model_path)
    state_dict = \
        torch.load(model_path, map_location=torch.device("cpu"))["model"]
    model.load_state_dict(state_dict)

    logger.info("Start evaluation using valid dataset")
    result = evaluate_synthesizer()
    logger.info(f"{result.metrics}")
    logger.info(f"generation rate: {result.generation_rate}")
    logger.info(f"generation time: {result.generation_time}")
    torch.save(result, result_path)

    logger.info("Copy log to output_dir")
    os.makedirs(output_dir, exist_ok=True)
    shutil.copyfile(result_path, os.path.join(output_dir, "result.pt"))
