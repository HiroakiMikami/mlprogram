import os
import pickle
from typing import Callable, List, Optional, TypeVar, cast

import numpy as np
import torch

from mlprogram import logging

V = TypeVar("V")

logger = logging.Logger(__name__)

groups = {}


def initialize(tmpdir: str,
               rank: Optional[int] = None, world_size: Optional[int] = None):
    if torch.distributed.is_initialized():
        logger.warning("torch.distributed is already initialized")
        return

    if rank is None and "RANK" not in os.environ:
        logger.warning("Abort torch.distributed.initialize (rank is not set)")
        return
    if world_size is None and "WORLD_SIZE" not in os.environ:
        logger.warning("Abort torch.distributed.initialize (world_size is not set)")
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
        group = group or torch.distributed.group.WORLD
        return cast(int, torch.distributed.get_rank(group=group))
    else:
        return 0


def size(group: Optional[torch.distributed.group] = None) -> int:
    if torch.distributed.is_initialized():
        group = group or torch.distributed.group.WORLD
        return cast(int, torch.distributed.get_world_size(group=group))
    else:
        return 1


def call(f: Callable, *args, **kwargs):
    if torch.distributed.is_initialized():
        return f(*args, **kwargs)


def all_gather(x: V,
               group: Optional[torch.distributed.group] = None,
               device=torch.device("cpu")) -> List[V]:
    if not torch.distributed.is_initialized():
        return [x]

    group = group or torch.distributed.group.WORLD
    data = pickle.dumps(x)
    length = [torch.zeros((), device=device, dtype=torch.long)
              for _ in range(size(group))]
    torch.distributed.all_gather(
        length,
        torch.tensor(len(data)),
        group=group
    )
    max_length = max(length)
    dst = [torch.zeros((max_length,), device=device, dtype=torch.uint8)
           for _ in range(size(group))]
    data += bytes(max_length - len(data))
    src = torch.from_numpy(np.frombuffer(data, dtype=np.uint8))
    torch.distributed.all_gather(
        dst,
        src,
        group=group
    )

    def load(length, data):
        return pickle.loads(data.cpu().numpy().tobytes()[:length])

    return [load(l_, d_) for l_, d_ in zip(length, dst)]
