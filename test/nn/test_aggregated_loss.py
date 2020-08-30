import torch
import unittest
from mlprogram.nn import AggregatedLoss


class TestAggregatedLoss(unittest.TestCase):
    def test_simple(self):
        loss = AggregatedLoss()
        val = loss(
            l0=torch.tensor(1, dtype=torch.float),
            l1=torch.tensor(10.0)
        )
        self.assertAlmostEqual(11.0, val.item())


if __name__ == "__main__":
    unittest.main()
