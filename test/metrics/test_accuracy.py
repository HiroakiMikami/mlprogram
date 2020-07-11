import unittest
from mlprogram.asts import Leaf
from mlprogram.metrics import Accuracy


def parse(value):
    return "parsed"


def unparse(value):
    return "Unparsed"


class TestAccuracy(unittest.TestCase):
    def test_simple_case(self):
        acc = Accuracy(parse, unparse)
        self.assertAlmostEqual(1.0, acc({"ground_truth": ["str"]}, "str"))

    def test_noramlize(self):
        metric = Accuracy(parse, unparse)
        self.assertAlmostEqual(1.0, metric(
            {"ground_truth": [Leaf("", "")]}, "Unparsed"))
        self.assertAlmostEqual(1.0, metric(
            {"ground_truth": ["Unparsed"]}, Leaf("", "")))
        self.assertAlmostEqual(1.0, metric(
            {"ground_truth": ["test"]}, "Unparsed"))
        self.assertAlmostEqual(1.0, metric(
            {"ground_truth": [Leaf("", "")]}, Leaf("", "")))


if __name__ == "__main__":
    unittest.main()
