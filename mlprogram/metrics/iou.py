import numpy as np
from typing import Dict, Any
from mlprogram.metrics import Metric


class Iou(Metric[np.array]):
    def __call__(self, input: Dict[str, Any],
                 actual: np.array) -> float:
        ground_truth = input["ground_truth"]
        iou = 0.0
        for gt in ground_truth:
            intersection = (gt & actual).astype(np.float).sum()
            union = (gt | actual).astype(np.float).sum()
            iou = max(iou, (intersection / union).item())
        return iou
