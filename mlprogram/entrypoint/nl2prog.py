import torch
from torch import nn
import json
from torch.utils.data import DataLoader
from typing \
    import Callable, Any, Optional, Tuple, cast, Iterable, List, Mapping
import os
import logging
import shutil
from math import ceil, isnan, isinf
from mlprogram.nn.utils.rnn import PaddedSequenceWithMask
from mlprogram.metrics import Metric
from mlprogram.synthesizers import Synthesizer
from mlprogram.utils import TopKModel
from mlprogram.utils.data import ListDataset
from mlprogram.utils.nl2prog import evaluate as eval, EvaluationResult


logger = logging.getLogger(__name__)


def train(workspace_dir: str, output_dir: str,
          dataset: torch.utils.data.Dataset,
          model: nn.Module,
          optimizer: torch.optim.Optimizer,
          loss,
          score,
          collate: Callable[[Any], Any],
          batch_size: int, num_epochs: int,
          num_checkpoints: int = 2, num_models: int = 3,
          device: torch.device = torch.device("cpu"),
          progress_bar: Optional[Callable[[Iterable], Iterable]] = None) \
        -> None:
    os.makedirs(workspace_dir, exist_ok=True)

    logger.info("Prepare model")
    model.to(device)
    model.train()

    # Load checkpoint
    checkpoint_dir = os.path.join(workspace_dir, "checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoints = [(os.path.join(checkpoint_dir, checkpoint),
                    int(checkpoint.replace(".pt", "")))
                   for checkpoint in os.listdir(checkpoint_dir)]
    checkpoints.sort(key=lambda x: x[1])
    if len(checkpoints) != 0:
        checkpoint_path = os.path.join(checkpoint_dir, checkpoints[-1][0])
        logger.info(f"Load checkpoint from {checkpoint_path}")
        ckpt = \
            torch.load(checkpoint_path, map_location=torch.device("cpu"))
        start_epoch = ckpt["epoch"]
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
    else:
        start_epoch = 0

    # Load log
    log_path = os.path.join(workspace_dir, "log.json")
    if os.path.exists(log_path):
        logger.info(f"Load log {log_path}")
        with open(log_path, "r") as file:
            logs = json.load(file)
    else:
        logs = []

    # Prepare TopKModel
    model_dir = os.path.join(workspace_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    top_k_model = TopKModel(num_models, model_dir)

    logger.info(f"Strat training from {start_epoch} epoch")
    for epoch in range(start_epoch, ceil(num_epochs)):
        # TODO num_workers > 0 causes the RuntimeError
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=0,
                            collate_fn=collate)
        if progress_bar is not None:
            loader = progress_bar(loader)
        if num_epochs - epoch < 1:
            n_iter = max(1, int(len(loader) * (num_epochs - epoch)))
        else:
            n_iter = -1

        avg_loss = 0.0
        avg_score = 0.0
        model.train()
        for i, batch in enumerate(loader):
            if len(batch) == 0:
                logger.info(f"Skip {i} th batch")
                continue
            if i == n_iter:
                break
            output = cast(Tuple[PaddedSequenceWithMask,
                                PaddedSequenceWithMask,
                                PaddedSequenceWithMask],
                          model(batch))
            bloss = loss(output)
            s = score(output)
            model.zero_grad()
            bloss.backward()
            optimizer.step()

            avg_loss += bloss.item() / len(loader)
            avg_score += s.item() / len(loader)

        logger.info(
            f"Epoch {epoch} : Loss = {avg_loss} Score = {avg_score}")
        logger.info("Save checkpoint")
        checkpoint = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        checkpoint_path = os.path.join(checkpoint_dir, f"{epoch}.pt")
        torch.save(checkpoint, checkpoint_path)
        checkpoints.append((checkpoint_path, epoch))
        while len(checkpoints) > num_checkpoints:
            path, _ = checkpoints.pop(0)
            logger.info(f"Remove {path}")
            os.remove(path)
        logger.info("Save log")
        logs.append({
            "epoch": epoch, "loss": avg_loss, "score": avg_score
        })
        with open(log_path, "w") as file:
            json.dump(logs, file)
        top_k_model.save(avg_score, f"{epoch}", model)

        if isnan(avg_loss) or isinf(avg_loss):
            logger.info("Stop training")
            break

    logger.info("Copy log to output_dir")
    os.makedirs(output_dir, exist_ok=True)
    shutil.copyfile(log_path, os.path.join(output_dir, "log.json"))

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
             n_samples: Optional[int] = None,
             progress_bar: Optional[Callable[[Iterable], Iterable]] = None) \
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

        test_data = test_dataset
        if progress_bar is not None:
            test_data = progress_bar(test_data)

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

        test_data = valid_dataset
        if progress_bar is not None:
            test_data = progress_bar(test_data)

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
