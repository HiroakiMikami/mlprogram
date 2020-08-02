import torch
from torch import nn
from tqdm import tqdm
from typing import Optional, Tuple, List, Mapping, Union
import os
import logging
import shutil
from mlprogram.metrics import Metric
from mlprogram.synthesizers import Synthesizer
from mlprogram.utils import evaluate as eval, EvaluationResult
from mlprogram.utils.data import ListDataset


logger = logging.getLogger(__name__)


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
    for name in os.listdir(model_dir):
        if name in results:
            continue
        path = os.path.join(model_dir, name)
        state_dict = \
            torch.load(path, map_location=torch.device("cpu"))["model"]
        logger.info(f"Start evaluation (test dataset): {name}")
        model.load_state_dict(state_dict)

        test_data = tqdm(test_dataset)

        result: EvaluationResult = eval(test_data,
                                        synthesizer,
                                        metrics=metrics, top_n=top_n)
        logger.info(f"{name}: {result.metrics}")
        results["test"][name] = result
        torch.save(results, results_path)

    logger.info("Find best model")
    best_model: Optional[str] = None
    if isinstance(main_metric, str):
        if main_metric == "generation":
            best_score: Union[float, Tuple[float, float]] = (-1.0, 0.0)
    else:
        best_score = -1.0
    for name, result in results["test"].items():
        if isinstance(main_metric, str):
            if main_metric == "generation":
                m = (result.generation_rate, -result.generation_time)
        else:
            m = result.metrics[main_metric[0]][main_metric[1]]
        if best_score < m:
            best_model = name
            best_score = m

    if best_model is not None:
        logger.info(f"Start evaluation (valid dataset): {best_model}")
        path = os.path.join(model_dir, best_model)
        state_dict = \
            torch.load(path, map_location=torch.device("cpu"))["model"]
        model.load_state_dict(state_dict)

        test_data = tqdm(valid_dataset)

        result = eval(test_data,
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
