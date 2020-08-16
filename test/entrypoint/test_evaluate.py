import unittest
import tempfile
import os
import torch

from mlprogram.entrypoint import evaluate
from mlprogram.utils.data import ListDataset
from mlprogram.metrics import Accuracy, Bleu
from mlprogram.entrypoint.evaluate import evaluate_synthesizer, Result
from mlprogram.synthesizers import Result as DecoderResult


class TestEvaluateSynthesizer(unittest.TestCase):
    def test_simple_case(self):
        def synthesize(input):
            query = input["input"]
            output = []
            if query == "query0":
                output = ["c0", "c1", "c2"]
            elif query == "query1":
                output = ["c2", "c3", "c0"]
            else:
                output = ["c2", "c3", "c5"]

            for i, s in enumerate(output):
                yield DecoderResult(s, -i, 1)

        accuracy = Accuracy()
        dataset = ListDataset([{
            "input": ["query0", "query1", "query2"],
            "ground_truth": ["c0", "c1", "c4"]
        }])
        results = evaluate_synthesizer(dataset, synthesize,
                                       metrics={"accuracy": accuracy})

        self.assertEqual(
            results.metrics,
            {1: {"accuracy": 1.0 / 3}, 3: {"accuracy": 2.0 / 3}})
        self.assertEqual(3, len(results.results))
        results.results[0].time = 0.0
        results.results[1].time = 0.0
        results.results[2].time = 0.0
        self.assertEqual(
            Result("query0", {"ground_truth": ["c0", "c1", "c4"]},
                   ["c0", "c1", "c2"],
                   {1: {"accuracy": 1.0}, 3: {"accuracy": 1.0}},
                   True, 0.0),
            results.results[0])
        self.assertEqual(
            Result("query1", {"ground_truth": ["c0", "c1", "c4"]},
                   ["c2", "c3", "c0"],
                   {1: {"accuracy": 0.0}, 3: {"accuracy": 1.0}},
                   True, 0.0),
            results.results[1])
        self.assertEqual(
            Result("query2", {"ground_truth": ["c0", "c1", "c4"]},
                   ["c2", "c3", "c5"],
                   {1: {"accuracy": 0.0}, 3: {"accuracy": 0.0}},
                   True, 0.0),
            results.results[2])


class TestEvaluate(unittest.TestCase):
    def prepare_dataset(self):
        return {"test": ListDataset([{"input": ["query"],
                                      "ground_truth": ["name0"]}]),
                "valid": ListDataset([{"input": ["query"],
                                       "ground_truth": ["name0"]}])}

    def prepare_model(self):
        class MockModel:
            def load_state_dict(self, state_dict):
                self.state_dict = state_dict

            def to(self, *args, **kwargs):
                pass

        return MockModel()

    def prepare_synthesizer(self, model):
        class MockSynthesizer:
            def __init__(self, model):
                self.model = model

            def __call__(self, query):
                yield DecoderResult(self.model.state_dict["name"],
                                    self.model.state_dict["score"],
                                    1)

        return MockSynthesizer(model)

    def test_happy_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input = os.path.join(tmpdir, "input")
            ws = os.path.join(tmpdir, "workspace")
            output = os.path.join(tmpdir, "output")
            os.makedirs(input)
            os.makedirs(os.path.join(input, "model"))
            torch.save({"model": {"score": 0.5, "name": "tmp"}},
                       os.path.join(input, "model", "0"))
            torch.save({"model": {"score": 1.0, "name": "tmp"}},
                       os.path.join(input, "model", "1"))
            dataset = self.prepare_dataset()
            model = self.prepare_model()
            evaluate(input, ws, output, dataset["test"], dataset["valid"],
                     model, self.prepare_synthesizer(model),
                     {
                "accuracy": Accuracy(),
                "bleu": Bleu(),
            }, (1, "bleu"))
            self.assertTrue(os.path.exists(os.path.join(ws, "results.pt")))
            results = torch.load(os.path.join(ws, "results.pt"))
            self.assertEqual(set(["test", "best_model", "valid"]),
                             set(results.keys()))
            self.assertEqual(set(["0", "1"]), set(results["test"].keys()))
            self.assertTrue(os.path.exists(
                os.path.join(output, "results.pt")))
            results = torch.load(os.path.join(output, "results.pt"))
            self.assertEqual(set(["test", "best_model", "valid"]),
                             set(results.keys()))
            self.assertEqual(set(["0", "1"]), set(results["test"].keys()))

    def test_resume(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input = os.path.join(tmpdir, "input")
            ws = os.path.join(tmpdir, "workspace")
            output = os.path.join(tmpdir, "output")
            os.makedirs(input)
            os.makedirs(os.path.join(input, "model"))
            torch.save({"model": {"score": 0.5, "name": "tmp"}},
                       os.path.join(input, "model", "0"))
            dataset = self.prepare_dataset()
            model = self.prepare_model()
            evaluate(input, ws, output, dataset["test"], dataset["valid"],
                     model, self.prepare_synthesizer(model),
                     {
                "accuracy": Accuracy(),
                "bleu": Bleu(),
            }, (1, "bleu"))
            self.assertTrue(os.path.exists(os.path.join(ws, "results.pt")))
            results = torch.load(os.path.join(ws, "results.pt"))
            self.assertEqual(set(["test", "best_model", "valid"]),
                             set(results.keys()))
            self.assertEqual(set(["0"]), set(results["test"].keys()))

            torch.save({"model": {"score": 1.0, "name": "tmp"}},
                       os.path.join(input, "model", "1"))
            dataset = self.prepare_dataset()
            evaluate(input, ws, output, dataset["test"], dataset["valid"],
                     model, self.prepare_synthesizer(model),
                     {
                "accuracy": Accuracy(),
                "bleu": Bleu(),
            }, (1, "bleu"))
            self.assertTrue(os.path.exists(os.path.join(ws, "results.pt")))
            results = torch.load(os.path.join(ws, "results.pt"))
            self.assertEqual(set(["test", "best_model", "valid"]),
                             set(results.keys()))
            self.assertEqual(set(["0", "1"]), set(results["test"].keys()))
            self.assertTrue(os.path.exists(
                os.path.join(output, "results.pt")))
            results = torch.load(os.path.join(output, "results.pt"))
            self.assertEqual(set(["test", "best_model", "valid"]),
                             set(results.keys()))
            self.assertEqual(set(["0", "1"]), set(results["test"].keys()))


if __name__ == "__main__":
    unittest.main()
