from typing import Dict

import torch

from mlprogram import logging

logger = logging.Logger(__name__)


def random_split(dataset: torch.utils.data.Dataset, ratio: Dict[str, float],
                 seed: int) \
        -> Dict[str, torch.utils.data.Dataset]:
    logger.info(f"Split dataset with seed={seed}")
    keys = list(ratio.keys())
    total = len(dataset)
    lengths = {key: int(total * ratio[key]) for key in keys}
    remain = total - sum([value for value in lengths.values()])
    lengths[keys[0]] += remain
    datasets = torch.utils.data.random_split(
        dataset,
        [lengths[key] for key in keys],
        generator=torch.Generator().manual_seed(seed))
    return {key: datasets[i] for i, key in enumerate(keys)}
