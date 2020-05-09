import unittest
from mlprogram.metrics import Accuracy
from mlprogram.utils import evaluate, Result
from mlprogram.utils.data import ListDataset


class TestEvaluate(unittest.TestCase):
    def test_simple_case(self):
        def synthesize(query):
            return "metadata", ["c0", "c1", "c2"]

        accuracy = Accuracy(str, str)
        dataset = \
            ListDataset([["query", ["c0"]], ["query", ["c1"]],
                         ["query", ["c4"]]])
        results = evaluate(dataset, synthesize, metrics={"accuracy": accuracy})

        self.assertEqual(
            results.metrics,
            {1: {"accuracy": 1.0 / 3}, 3: {"accuracy": 2.0 / 3}})
        self.assertEqual(3, len(results.results))
        self.assertEqual(
            Result("query", ["c0"], "metadata", ["c0", "c1", "c2"],
                   {1: {"accuracy": 1.0}, 3: {"accuracy": 1.0}}),
            results.results[0])
        self.assertEqual(
            Result("query", ["c1"], "metadata", ["c0", "c1", "c2"],
                   {1: {"accuracy": 0.0}, 3: {"accuracy": 1.0}}),
            results.results[1])
        self.assertEqual(
            Result("query", ["c4"], "metadata", ["c0", "c1", "c2"],
                   {1: {"accuracy": 0.0}, 3: {"accuracy": 0.0}}),
            results.results[2])


if __name__ == "__main__":
    unittest.main()
