import os
from typing import Callable, Optional, cast

import torch

from mlprogram import logging

logger = logging.Logger(__name__)

groups = {}


def initialize(tmpdir: str,
               rank: Optional[int] = None, world_size: Optional[int] = None):
    if rank is None and "RANK" not in os.environ:
        logger.warning(
            "Abort torch.distributed.initialization (rank is not set)")
        return
    if world_size is None and "WORLD_SIZE" not in os.environ:
        logger.warning(
            "Abort torch.distributed.initialization (world_size is not set)")
        return

    if rank is None:
        rank = int(os.environ["RANK"])
    if world_size is None:
        world_size = int(os.environ["WORLD_SIZE"])
    use_env = "MASTER_ADDR" in os.environ and "MASTER_PORT" in os.environ
    if use_env:
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        init_method = "env://"
    else:
        path = os.path.join(tmpdir, "distributed")
        init_method = f"file://{path}"
    backend = "gloo"
    logger.info(f"Initialize torch.distributed with backend={backend}")
    torch.distributed.init_process_group(
        backend, rank=rank, world_size=world_size,
        init_method=init_method
    )

    logger.info("Create new gloo group")
    groups["world_gloo"] = torch.distributed.new_group(
        list(range(world_size)), backend="gloo"
    )
    if torch.cuda.is_available():
        logger.info("Create new NCCL group")
        groups["world_nccl"] = torch.distributed.new_group(
            list(range(world_size)), backend="nccl"
        )


def is_initialized() -> bool:
    return cast(bool, torch.distributed.is_initialized())


def is_main_process() -> bool:
    if torch.distributed.is_initialized():
        return cast(int, torch.distributed.get_rank()) == 0
    else:
        return True


def rank(group: Optional[torch.distributed.group] = None) -> int:
    if torch.distributed.is_initialized():
        return cast(int, torch.distributed.get_rank(group=group))
    else:
        return 0


def size(group: Optional[torch.distributed.group] = None) -> int:
    if torch.distributed.is_initialized():
        return cast(int, torch.distributed.get_world_size(group=group))
    else:
        return 1


def call(f: Callable, *args, **kwargs):
    if torch.distributed.is_initialized():
        return f(*args, **kwargs)
