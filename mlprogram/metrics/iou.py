import numpy as np
from typing import Dict, Any
from mlprogram.metrics import Metric


class Iou(Metric[np.array]):
    def __call__(self, input: Dict[str, Any],
                 actual: np.array) -> float:
        ground_truth = input["ground_truth"]

        if ground_truth.sum() == 0:
            iou = 1.0 - actual.sum() / np.prod(actual.shape)
        else:
            intersection = (ground_truth & actual).astype(np.float).sum()
            union = (ground_truth | actual).astype(np.float).sum()
            iou = (intersection / union).item()
        return iou
