import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pytorch_pfn_extras as ppe
from pytorch_pfn_extras.training import extensions
from typing \
    import Callable, Any, Optional, Tuple, cast, List, Mapping
import os
import logging
import shutil
from math import isnan, isinf
from mlprogram.nn.utils.rnn import PaddedSequenceWithMask
from mlprogram.metrics import Metric
from mlprogram.synthesizers import Synthesizer
from mlprogram.utils import TopKModel
from mlprogram.utils.data import ListDataset
from mlprogram.utils.nl2prog import evaluate as eval, EvaluationResult


logger = logging.getLogger(__name__)


class Trigger:
    def __init__(self, interval: int, n_iter: int):
        self.interval = interval
        self.n_iter = n_iter

    def __call__(self, manager):
        return (manager.updater.iteration == self.n_iter) or \
            (manager.updater.iteration % self.interval == 0)


def train(workspace_dir: str, output_dir: str,
          dataset: torch.utils.data.Dataset,
          model: nn.Module,
          optimizer: torch.optim.Optimizer,
          loss,
          score,
          collate: Callable[[Any], Any],
          batch_size: int, num_epochs: int,
          num_checkpoints: int = 2, num_models: int = 3,
          device: torch.device = torch.device("cpu")) \
        -> None:
    os.makedirs(workspace_dir, exist_ok=True)

    logger.info("Prepare model")
    model.to(device)
    model.train()

    # Initialize extensions manager
    iter_per_epoch = len(dataset) // batch_size
    n_iter = max(1, (len(dataset) * num_epochs) // batch_size)
    log_reporter = \
        extensions.LogReport(trigger=Trigger(iter_per_epoch, n_iter))
    manager = ppe.training.ExtensionsManager(
        model, optimizer, num_epochs,
        out_dir=workspace_dir,
        extensions=[
            log_reporter,
            extensions.ProgressBar(),
            extensions.PrintReport(),
        ],
        iters_per_epoch=iter_per_epoch,
    )
    snapshot = extensions.snapshot(autoload=True, n_retains=1)
    manager.extend(snapshot, trigger=Trigger(iter_per_epoch, n_iter))

    # Prepare TopKModel
    model_dir = os.path.join(workspace_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    top_k_model = TopKModel(num_models, model_dir)

    epoch = manager.updater.iteration // iter_per_epoch
    while True:
        # TODO num_workers > 0 causes the RuntimeError
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=0,
                            collate_fn=collate)
        model.train()
        epoch_loss = 0.0
        epoch_score = 0.0
        for i, batch in enumerate(loader):
            if manager.updater.iteration >= n_iter:
                break
            with manager.run_iteration():
                if len(batch) == 0:
                    logger.info(f"Skip {i} th batch")
                    continue
                output = cast(Tuple[PaddedSequenceWithMask,
                                    PaddedSequenceWithMask,
                                    PaddedSequenceWithMask],
                              model(batch))
                bloss = loss(output)
                s = score(output)
                model.zero_grad()
                bloss.backward()
                optimizer.step()

                ppe.reporting.report({
                    "loss": bloss.item(),
                    "score": s.item()
                })
            if len(log_reporter.log) != 0:
                epoch_loss = log_reporter.log[-1]["loss"]
                epoch_score = log_reporter.log[-1]["score"]

        top_k_model.save(epoch_score, f"{epoch}", model)

        if isnan(epoch_loss) or isinf(epoch_loss):
            logger.info("Stop training")
            break
        if manager.updater.iteration >= n_iter:
            break
        epoch += 1

    logger.info("Copy log to output_dir")
    os.makedirs(output_dir, exist_ok=True)
    shutil.copyfile(os.path.join(workspace_dir, "log"),
                    os.path.join(output_dir, "log.json"))

    logger.info("Copy models to output_dir")
    out_model_dir = os.path.join(output_dir, "model")
    if os.path.exists(out_model_dir):
        shutil.rmtree(out_model_dir)
    shutil.copytree(model_dir, out_model_dir)


def evaluate(input_dir: str, workspace_dir: str, output_dir: str,
             test_dataset: torch.utils.data.Dataset,
             valid_dataset: torch.utils.data.Dataset,
             synthesizer: Synthesizer,
             metrics: Mapping[str, Metric],
             main_metric: Tuple[int, str],
             top_n: List[int] = [1],
             device: torch.device = torch.device("cpu"),
             n_samples: Optional[int] = None) \
        -> None:
    os.makedirs(workspace_dir, exist_ok=True)

    if n_samples is not None:
        test_dataset = ListDataset(test_dataset[:n_samples])
        valid_dataset = ListDataset(valid_dataset[:n_samples])

    logger.info("Prepare synthesizer")
    synthesizer.to(device)

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
        synthesizer.load_state_dict(state_dict)

        test_data = tqdm(test_dataset)

        result: EvaluationResult = eval(test_data,
                                        synthesizer,
                                        metrics=metrics, top_n=top_n)
        logger.info(f"{name}: {result.metrics}")
        results["test"][name] = result
        torch.save(results, results_path)

    logger.info("Find best model")
    best_model: Optional[str] = None
    best_score: float = -1.0
    for name, result in results["test"].items():
        m = result.metrics[main_metric[0]][main_metric[1]]
        if best_score < m:
            best_model = name
            best_score = m

    if best_model is not None:
        logger.info(f"Start evaluation (valid dataset): {best_model}")
        path = os.path.join(model_dir, best_model)
        state_dict = \
            torch.load(path, map_location=torch.device("cpu"))["model"]
        synthesizer.load_state_dict(state_dict)

        test_data = tqdm(valid_dataset)

        result = eval(test_data,
                      synthesizer,
                      metrics=metrics, top_n=top_n)
        logger.info(f"{name}: {result.metrics}")
        results["best_model"] = best_model
        results["valid"] = result
        torch.save(results, results_path)

    logger.info("Copy log to output_dir")
    os.makedirs(output_dir, exist_ok=True)
    shutil.copyfile(results_path, os.path.join(output_dir, "results.pt"))
