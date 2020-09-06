import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import pytorch_pfn_extras as ppe
from pytorch_pfn_extras.training import extensions
from typing import Callable, Any, Union, Optional, List
import os
import shutil
from math import isnan, isinf
from mlprogram.utils import TopKModel
from mlprogram.metrics import Metric
from mlprogram.synthesizers import Synthesizer
from mlprogram.utils import logging
from mlprogram import distributed
from dataclasses import dataclass

logger = logging.Logger(__name__)


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


def calc_n_iter(length: Length, iter_per_epoch: int) -> int:
    if isinstance(length, Epoch):
        n_iter = length.n * iter_per_epoch
    else:
        n_iter = length.n
    return n_iter


def calc_interval_iter(interval: Optional[Length], iter_per_epoch: int) -> int:
    if interval is None:
        interval_iter = iter_per_epoch
    else:
        if isinstance(interval, Epoch):
            interval_iter = interval.n * iter_per_epoch
        else:
            interval_iter = interval.n
    return interval_iter


def create_extensions_manager(n_iter: int, interval_iter: int,
                              iter_per_epoch: int,
                              model: nn.Module,
                              optimizer: torch.optim.Optimizer,
                              workspace_dir: str):
    logger.info("Prepare pytorch-pfn-extras")
    manager = ppe.training.ExtensionsManager(
        model, optimizer, n_iter / iter_per_epoch,
        out_dir=workspace_dir,
        extensions=[],
        iters_per_epoch=iter_per_epoch,
    )
    if distributed.is_main_process():
        log_reporter = \
            extensions.LogReport(trigger=Trigger(interval_iter, n_iter))
        manager.extend(log_reporter)
        manager.extend(extensions.ProgressBar())
        manager.extend(extensions.PrintReport(entries=[
            "loss", "score",
            "iteration", "epoch",
            "time.iteration", "gpu.time.iteration", "elapsed_time"
        ]),
                       trigger=Trigger(interval_iter, n_iter))
    else:
        log_reporter = None
    if distributed.is_initialized():
        snapshot = extensions.snapshot(autoload=True, n_retains=1,
                                       saver_rank=0)
        snapshot._rank = distributed.rank()
        snapshot._size = distributed.size()
        snapshot._local_rank = distributed.rank()
    else:
        snapshot = extensions.snapshot(autoload=True, n_retains=1)
    manager.extend(snapshot, trigger=Trigger(interval_iter, n_iter))
    return log_reporter, manager


def process_group(device: torch.device) -> Optional[torch.distributed.group]:
    if not distributed.is_initialized():
        return None
    else:
        if device.type == "cuda":
            return distributed.groups["world_nccl"]
        else:
            return distributed.groups["world_gloo"]


class StopTrainingException(Exception):
    pass


def abort_if_loss_is_nan(log_reporter):
    stop = torch.tensor(0)
    if distributed.is_main_process():
        if len(log_reporter.log) != 0:
            log_loss = log_reporter.log[-1]["loss"]

            if isnan(log_loss) or isinf(log_loss):
                logger.info("Stop training")
                stop = torch.tensor(1)
    if distributed.is_initialized():
        torch.distributed.broadcast(stop, src=0,
                                    group=distributed.groups["world_gloo"])

    if stop.item() == 1:
        raise StopTrainingException()


def all_reduce(model: nn.Module, group: torch.distributed.group):
    nparams = list(model.named_parameters())
    nparams.sort(key=lambda x: x[0])
    params = [p.grad if p.grad is not None else torch.zeros_like(p.data)
              for _, p in nparams]
    size = torch.distributed.get_world_size(group)
    torch.distributed.all_reduce_coalesced(
        params, group=group
    )
    for p in params:
        p /= size


def train_supervised(workspace_dir: str, output_dir: str,
                     dataset: torch.utils.data.Dataset,
                     model: nn.Module,
                     optimizer: torch.optim.Optimizer,
                     loss: Callable[[Any], torch.Tensor],
                     score: Callable[[Any], torch.Tensor],
                     collate: Callable[[List[Any]], Any],
                     batch_size: int,
                     length: Length,
                     interval: Optional[Length] = None,
                     num_models: int = 3,
                     device: torch.device = torch.device("cpu")) \
        -> None:
    os.makedirs(workspace_dir, exist_ok=True)

    logger.info("Prepare model")
    model.to(device)
    model.train()

    group = process_group(device)

    if hasattr(dataset, "__len__"):
        is_iterable = False
        iter_per_epoch = len(dataset) // batch_size
    else:
        is_iterable = True
        iter_per_epoch = 1
    n_iter = calc_n_iter(length, iter_per_epoch)
    interval_iter = calc_interval_iter(interval, iter_per_epoch)

    # Initialize extensions manager
    log_reporter, manager = \
        create_extensions_manager(n_iter, interval_iter, iter_per_epoch,
                                  model, optimizer, workspace_dir)

    # Prepare TopKModel
    model_dir = os.path.join(workspace_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    top_k_model = TopKModel(num_models, model_dir)

    log_score = 0.0
    logger.info("Start training")
    try:
        while manager.updater.iteration < n_iter:
            # TODO num_workers > 0 causes the RuntimeError
            if is_iterable:
                loader = DataLoader(dataset, batch_size=batch_size,
                                    shuffle=False, num_workers=0,
                                    collate_fn=collate)
            else:
                loader = DataLoader(dataset, batch_size=batch_size,
                                    shuffle=True, num_workers=0,
                                    collate_fn=collate)
            model.train()
            for batch in logger.iterable_block("iteration", loader, True):
                if manager.updater.iteration >= n_iter:
                    break
                if len(batch) == 0:
                    logger.warning(
                        f"Skip {manager.updater.iteration} th batch")
                    continue
                with manager.run_iteration():
                    with logger.block("forward"):
                        output = model(batch)
                        bloss = loss(output)
                        s = score(output)
                    with logger.block("backward"):
                        model.zero_grad()
                        bloss.backward()
                    if group is not None:
                        with logger.block("all-reduce"):
                            all_reduce(model, group)
                    with logger.block("optimizer.step"):
                        optimizer.step()

                    ppe.reporting.report({
                        "loss": bloss.item(),
                        "score": s.item()
                    })
                    logger.dump_eplased_time_log()
                    if device.type == "cuda":
                        ppe.reporting.report({
                            "gpu.max_memory_allocated":
                                torch.cuda.max_memory_allocated(device)
                        })

                if distributed.is_main_process():
                    if len(log_reporter.log) != 0:
                        log_score = log_reporter.log[-1]["score"]
                    if manager.updater.iteration % interval_iter == 0:
                        logger.debug("Update top-K model: score={log_score}")
                        top_k_model.save(log_score,
                                         f"{manager.updater.iteration}", model)

                abort_if_loss_is_nan(log_reporter)
    except StopTrainingException:  # noqa
        pass

    logger.info("Copy log to output_dir")
    os.makedirs(output_dir, exist_ok=True)
    shutil.copyfile(os.path.join(workspace_dir, "log"),
                    os.path.join(output_dir, "log.json"))

    logger.info("Copy models to output_dir")
    out_model_dir = os.path.join(output_dir, "model")
    if os.path.exists(out_model_dir):
        shutil.rmtree(out_model_dir)
    shutil.copytree(model_dir, out_model_dir)

    logger.info("Dump the last model")
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
    torch.save(optimizer.state_dict(),
               os.path.join(output_dir, "optimizer.pt"))


def train_REINFORCE(input_dir: str, workspace_dir: str, output_dir: str,
                    dataset: torch.utils.data.Dataset,
                    synthesizer: Synthesizer,
                    model: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    loss: Callable[[Any], torch.Tensor],
                    score: Metric,
                    reward: Callable[[float], float],
                    rollout_transform: Callable[[Any], Any],
                    collate: Callable[[List[Any]], Any],
                    batch_size: int,
                    n_rollout: int,
                    length: Length,
                    interval: Optional[Length] = None,
                    num_models: int = 3,
                    use_pretrained_model: bool = False,
                    use_pretrained_optimizer: bool = False,
                    device: torch.device = torch.device("cpu")) \
        -> None:
    os.makedirs(workspace_dir, exist_ok=True)

    logger.info("Prepare model")
    model.to(device)
    model.train()

    group = process_group(device)

    if hasattr(dataset, "__len__"):
        is_iterable = False
        iter_per_epoch = len(dataset) // batch_size
    else:
        is_iterable = True
        iter_per_epoch = 1
    n_iter = calc_n_iter(length, iter_per_epoch)
    interval_iter = calc_interval_iter(interval, iter_per_epoch)

    if use_pretrained_model:
        logger.info("Load pretrained model")
        pretrained_model = os.path.join(input_dir, "model.pt")
        state_dict = torch.load(pretrained_model,
                                map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
    if use_pretrained_optimizer:
        logger.info("Load pretrained optimizer")
        pretrained_optimizer = os.path.join(input_dir, "optimizer.pt")
        state_dict = torch.load(pretrained_optimizer,
                                map_location=torch.device("cpu"))
        optimizer.load_state_dict(state_dict)

    # Initialize extensions manager
    log_reporter, manager = \
        create_extensions_manager(n_iter, interval_iter, iter_per_epoch,
                                  model, optimizer, workspace_dir)

    # Prepare TopKModel
    model_dir = os.path.join(workspace_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    top_k_model = TopKModel(num_models, model_dir)

    log_score = 0.0
    logger.info("Start training")
    try:
        while manager.updater.iteration < n_iter:
            # TODO num_workers > 0 causes the RuntimeError
            if is_iterable:
                loader = DataLoader(dataset, batch_size=batch_size,
                                    shuffle=False, num_workers=0,
                                    collate_fn=lambda x: x)
            else:
                loader = DataLoader(dataset, batch_size=batch_size,
                                    shuffle=True, num_workers=0,
                                    collate_fn=lambda x: x)
            model.train()
            for samples in logger.iterable_block("iteration", loader, True):
                if manager.updater.iteration >= n_iter:
                    break
                # Rollout
                rollouts = []
                scores = []
                with torch.no_grad():
                    for sample in logger.iterable_block("rollout", samples):
                        input = rollout_transform(sample)
                        for rollout in logger.iterable_block(
                                "sample",
                                synthesizer(input,
                                            n_required_output=n_rollout)):
                            if not rollout.is_finished:
                                continue
                            s = score(sample, rollout.output)
                            for _ in range(rollout.num):
                                output = {key: value
                                          for key, value in input.items()}
                                output["ground_truth"] = rollout.output
                                output["reward"] = torch.tensor(reward(s))
                                rollouts.append(output)
                                scores.append(s)
                if len(rollouts) == 0:
                    logger.warning("No rollout")
                    continue
                if len(rollouts) != n_rollout:
                    logger.warning(
                        "#rollout is unexpected: "
                        f"expected={n_rollout} actual={len(rollouts)}")

                with manager.run_iteration():
                    with logger.block("collate"):
                        batch2 = collate(rollouts)
                    with logger.block("forward"):
                        model.train()
                        output = model(batch2)
                        bloss = loss(output)
                    with logger.block("backward"):
                        model.zero_grad()
                        bloss.backward()
                    if group is not None:
                        with logger.block("all-reduce"):
                            all_reduce(model, group)
                    with logger.block("optimizer.step"):
                        optimizer.step()

                    ppe.reporting.report({
                        "loss": bloss.item(),
                        "score": np.mean(scores)
                    })
                    logger.dump_eplased_time_log()
                    if device.type == "cuda":
                        ppe.reporting.report({
                            "gpu.max_memory_allocated":
                                torch.cuda.max_memory_allocated(device)
                        })

                if distributed.is_main_process():
                    if len(log_reporter.log) != 0:
                        log_score = log_reporter.log[-1]["score"]
                    if manager.updater.iteration % interval_iter == 0:
                        logger.debug("Update top-K model: score={log_score}")
                        top_k_model.save(log_score,
                                         f"{manager.updater.iteration}", model)

                abort_if_loss_is_nan(log_reporter)
    except StopTrainingException:  # noqa
        pass

    logger.info("Copy log to output_dir")
    os.makedirs(output_dir, exist_ok=True)
    shutil.copyfile(os.path.join(workspace_dir, "log"),
                    os.path.join(output_dir, "log.json"))

    logger.info("Copy models to output_dir")
    out_model_dir = os.path.join(output_dir, "model")
    if os.path.exists(out_model_dir):
        shutil.rmtree(out_model_dir)
    shutil.copytree(model_dir, out_model_dir)

    logger.info("Dump the last model")
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
    torch.save(optimizer.state_dict(),
               os.path.join(output_dir, "optimizer.pt"))
