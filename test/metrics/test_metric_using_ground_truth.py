import unittest
from mlprogram.asts import Leaf
from mlprogram.metrics.metric_using_ground_truth import MetricUsingGroundTruth


class MockMetric(MetricUsingGroundTruth):
    def __init__(self, parse, unparse):
        super().__init__(parse, unparse)

    def metric(self, gts, value):
        return sum([int(gt) for gt in gts]) + int(value)


class TestMetric(unittest.TestCase):
    def test_noramlize(self):
        metric = MockMetric(lambda x: Leaf("int", str(int(x) + 1)),
                            lambda x: str(x.value))
        self.assertAlmostEqual(5.0, metric({"ground_truth": ["1"]},
                                           Leaf("", "2")))


if __name__ == "__main__":
    unittest.main()
