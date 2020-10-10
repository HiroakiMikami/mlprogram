import numpy as np
import torch
from mlprogram.nn import AggregatedLoss


class TestAggregatedLoss(object):
    def test_simple(self):
        loss = AggregatedLoss()
        val = loss(
            l0=torch.tensor(1, dtype=torch.float),
            l1=torch.tensor(10.0)
        )
        assert np.allclose(11.0, val.item())
