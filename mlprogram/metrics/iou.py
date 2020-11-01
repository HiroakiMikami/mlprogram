import numpy as np

from mlprogram import Environment
from mlprogram.metrics.metric import Metric


class Iou(Metric[np.array]):
    def __call__(self, input: Environment,
                 actual: np.array) -> float:
        ground_truth = input.supervisions["ground_truth"]

        if ground_truth.sum() == 0:
            iou = 1.0 - actual.sum() / np.prod(actual.shape)
        else:
            intersection = (ground_truth & actual).astype(np.float).sum()
            union = (ground_truth | actual).astype(np.float).sum()
            iou = (intersection / union).item()
        return iou
