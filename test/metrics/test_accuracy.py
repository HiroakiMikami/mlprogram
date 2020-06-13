import unittest
from mlprogram.asts import Leaf
from mlprogram.metrics import Accuracy


def parse(value):
    return "parsed"


def unparse(value):
    return "unparsed"


class TestAccuracy(unittest.TestCase):
    def test_simple_case(self):
        acc = Accuracy(parse, unparse)
        self.assertAlmostEqual(1.0, acc(["str"], "str"))
        self.assertAlmostEqual(0.0, acc(["str"], ""))

    def test_noramlize(self):
        metric = Accuracy(parse, unparse)
        self.assertAlmostEqual(1.0, metric([Leaf("", "")], "unparsed"))
        self.assertAlmostEqual(1.0, metric(["unparsed"], Leaf("", "")))
        self.assertAlmostEqual(1.0, metric(["test"], "unparsed"))
        self.assertAlmostEqual(1.0, metric([Leaf("", "")], Leaf("", "")))


if __name__ == "__main__":
    unittest.main()
