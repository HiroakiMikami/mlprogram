import numpy as np
from torch import nn


class Iou(nn.Module):
    def forward(self, expected: np.array, actual: np.array) -> float:
        if expected.sum() == 0:
            iou = float(1.0 - actual.sum() / np.prod(actual.shape))
        else:
            intersection = (expected & actual).astype(np.float).sum()
            union = (expected | actual).astype(np.float).sum()
            iou = (intersection / union).item()
        return iou
