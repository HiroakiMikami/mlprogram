import time
from dataclasses import dataclass
from typing \
    import List, Dict, TypeVar, Generic, Mapping, Any, Optional, Tuple, Union
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import os
import logging
import shutil
from mlprogram.metrics import Metric
from mlprogram.synthesizers import Synthesizer
from mlprogram.utils.data import ListDataset


logger = logging.getLogger(__name__)


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


def evaluate_synthesizer(dataset: torch.utils.data.Dataset,
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
            logger.debug("start evaluation")
            n_query += 1
            begin = time.time()
            candidates = list(synthesizer({"input": input}))
            end = time.time()
            logger.debug("calculate metrics")
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


def evaluate(input_dir: str, workspace_dir: str, output_dir: str,
             test_dataset: torch.utils.data.Dataset,
             valid_dataset: torch.utils.data.Dataset,
             model: nn.Module,
             synthesizer: Synthesizer,
             metrics: Mapping[str, Metric],
             main_metric: Union[Tuple[int, str], str],
             top_n: List[int] = [1],
             device: torch.device = torch.device("cpu"),
             n_samples: Optional[int] = None) \
        -> None:
    if isinstance(main_metric, str):
        assert main_metric == "generation"
    os.makedirs(workspace_dir, exist_ok=True)

    if n_samples is not None:
        test_dataset = ListDataset(test_dataset[:n_samples])
        valid_dataset = ListDataset(valid_dataset[:n_samples])

    logger.info("Prepare model")
    model.to(device)

    model_dir = os.path.join(input_dir, "model")
    results_path = os.path.join(workspace_dir, "results.pt")
    if os.path.exists(results_path):
        logger.info(f"Load results from {results_path}")
        results = torch.load(results_path)
    else:
        results = {"test": {}}
    logger.info("Find the best model using test dataset")
    for name in os.listdir(model_dir):
        if name in results:
            continue
        path = os.path.join(model_dir, name)
        state_dict = \
            torch.load(path, map_location=torch.device("cpu"))["model"]
        logger.info(f"Start evaluation (test dataset): {name}")
        model.load_state_dict(state_dict)

        test_data = tqdm(test_dataset)

        result: EvaluationResult = evaluate_synthesizer(
            test_data, synthesizer, metrics=metrics, top_n=top_n)
        logger.info(f"{name}: {result.metrics}")
        results["test"][name] = result
        torch.save(results, results_path)

    logger.info("Find best model")
    best_model: Optional[str] = None
    if isinstance(main_metric, str):
        if main_metric == "generation":
            best_score0 = (-1.0, 0.0)
            for name, result in results["test"].items():
                m0 = (result.generation_rate, -result.generation_time)
            if best_score0 < m0:
                best_model = name
                best_score0 = m0
    else:
        best_score1 = -1.0
        for name, result in results["test"].items():
            m1 = result.metrics[main_metric[0]][main_metric[1]]
            if best_score1 < m1:
                best_model = name
                best_score1 = m1

    if best_model is not None:
        logger.info(f"Start evaluation (valid dataset): {best_model}")
        path = os.path.join(model_dir, best_model)
        state_dict = \
            torch.load(path, map_location=torch.device("cpu"))["model"]
        model.load_state_dict(state_dict)

        test_data = tqdm(valid_dataset)

        logger.info("Start evaluation using valid dataset")
        result = evaluate_synthesizer(test_data,
                                      synthesizer,
                                      metrics=metrics, top_n=top_n)
        logger.info(f"{name}: {result.metrics}")
        logger.info(f"generation rate: {result.generation_rate}")
        logger.info(f"generation time: {result.generation_time}")
        results["best_model"] = best_model
        results["valid"] = result
        torch.save(results, results_path)

    logger.info("Copy log to output_dir")
    os.makedirs(output_dir, exist_ok=True)
    shutil.copyfile(results_path, os.path.join(output_dir, "results.pt"))
