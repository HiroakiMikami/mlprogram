import unittest
from mlprogram.ast import Leaf
from mlprogram.metrics import Metric


class TestMetric(unittest.TestCase):
    def test_noramlize(self):
        def metric(gts, value):
            return sum([int(gt) for gt in gts]) + int(value)

        metric = Metric(lambda x: Leaf("int", str(int(x) + 1)),
                        lambda x: str(x.value),
                        metric)
        self.assertAlmostEqual(5.0, metric(["1"], "2"))
        self.assertAlmostEqual(3.0, metric([Leaf("", "1")], "2"))
        self.assertAlmostEqual(3.0, metric([Leaf("", "1")], Leaf("", "2")))


if __name__ == "__main__":
    unittest.main()
