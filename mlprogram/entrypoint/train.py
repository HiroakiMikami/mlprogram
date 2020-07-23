import torch
from torch import nn
from torch.utils.data import DataLoader
import pytorch_pfn_extras as ppe
from pytorch_pfn_extras.training import extensions
from typing import Callable, Any, Tuple, cast, Union, Optional
import os
import logging
import shutil
from math import isnan, isinf
from mlprogram.nn.utils.rnn import PaddedSequenceWithMask
from mlprogram.utils import TopKModel
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Epoch:
    n: int


@dataclass
class Iteration:
    n: int


Length = Union[Epoch, Iteration]


class Trigger:
    def __init__(self, interval: int, n_iter: int):
        self.interval = interval
        self.n_iter = n_iter

    def __call__(self, manager):
        return (manager.updater.iteration == self.n_iter) or \
            (manager.updater.iteration % self.interval == 0)


def train_supervised(workspace_dir: str, output_dir: str,
                     dataset: torch.utils.data.Dataset,
                     model: nn.Module,
                     optimizer: torch.optim.Optimizer,
                     loss,
                     score,
                     collate: Callable[[Any], Any],
                     batch_size: int,
                     length: Length,
                     interval: Optional[Length] = None,
                     num_checkpoints: int = 2, num_models: int = 3,
                     device: torch.device = torch.device("cpu")) \
        -> None:
    os.makedirs(workspace_dir, exist_ok=True)

    logger.info("Prepare model")
    model.to(device)
    model.train()

    iter_per_epoch = len(dataset) // batch_size
    if isinstance(length, Epoch):
        n_iter = length.n * iter_per_epoch
    else:
        n_iter = length.n
    if interval is None:
        interval_iter = iter_per_epoch
    else:
        if isinstance(interval, Epoch):
            interval_iter = interval.n * iter_per_epoch
        else:
            interval_iter = interval.n

    # Initialize extensions manager
    log_reporter = \
        extensions.LogReport(trigger=Trigger(interval_iter, n_iter))
    manager = ppe.training.ExtensionsManager(
        model, optimizer, n_iter / iter_per_epoch,
        out_dir=workspace_dir,
        extensions=[
            log_reporter,
            extensions.ProgressBar(),
        ],
        iters_per_epoch=iter_per_epoch,
    )
    manager.extend(extensions.PrintReport(),
                   trigger=Trigger(interval_iter, n_iter))
    snapshot = extensions.snapshot(autoload=True, n_retains=1)
    manager.extend(snapshot, trigger=Trigger(interval_iter, n_iter))

    # Prepare TopKModel
    model_dir = os.path.join(workspace_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    top_k_model = TopKModel(num_models, model_dir)

    log_loss = 0.0
    log_score = 0.0
    while manager.updater.iteration < n_iter:
        # TODO num_workers > 0 causes the RuntimeError
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=0,
                            collate_fn=collate)
        model.train()
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
                log_loss = log_reporter.log[-1]["loss"]
                log_score = log_reporter.log[-1]["score"]

            if manager.updater.iteration % interval_iter == 0:
                top_k_model.save(log_score,
                                 f"{manager.updater.iteration}", model)

            if isnan(log_loss) or isinf(log_loss):
                logger.info("Stop training")
                break

    logger.info("Copy log to output_dir")
    os.makedirs(output_dir, exist_ok=True)
    shutil.copyfile(os.path.join(workspace_dir, "log"),
                    os.path.join(output_dir, "log.json"))

    logger.info("Copy models to output_dir")
    out_model_dir = os.path.join(output_dir, "model")
    if os.path.exists(out_model_dir):
        shutil.rmtree(out_model_dir)
    shutil.copytree(model_dir, out_model_dir)
