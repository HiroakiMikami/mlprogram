import torch
import unittest
from mlprogram.nn import AggregatedLoss


class TestAggregatedLoss(unittest.TestCase):
    def test_simple(self):
        loss = AggregatedLoss({
            "l0": lambda x: torch.tensor(len(x), dtype=torch.float),
            "l1": lambda x: torch.tensor(10.0)
        })
        val = loss({"in": None})
        self.assertAlmostEqual(11.0, val.item())


if __name__ == "__main__":
    unittest.main()
