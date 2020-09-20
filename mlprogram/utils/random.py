import torch
import numpy as np
from typing import List, Dict
from mlprogram.utils import logging

logger = logging.Logger(__name__)


def split(rng: np.random.RandomState, n: int, k: int, eps: float) -> List[int]:
    probs = [1 / (n + 1) - eps for _ in range(n + 2)]
    divisors: List[int] = []
    for i, m in enumerate(rng.multinomial(k - 1, probs)):
        for _ in range(m):
            divisors.append(i)
    last = 0
    ks = []
    for divisor in divisors:
        ks.append(divisor - last)
        last = divisor
    ks.append(n - last)
    return ks


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
