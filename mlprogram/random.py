import numpy as np
from typing import List
from mlprogram import logging

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
