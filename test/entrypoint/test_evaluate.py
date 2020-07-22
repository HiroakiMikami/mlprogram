import unittest
import tempfile
import os
import torch

from mlprogram.entrypoint import evaluate
from mlprogram.asts import Leaf
from mlprogram.utils.data import ListDataset
from mlprogram.synthesizers import Result
from mlprogram.metrics import Accuracy, Bleu


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
                yield Result(Leaf("str", self.model.state_dict["name"]),
                             self.model.state_dict["score"])

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
                "accuracy": Accuracy(lambda x: Leaf("str", x),
                                     lambda x: x.value),
                "bleu": Bleu(lambda x: Leaf("str", x),
                             lambda x: x.value),
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
                "accuracy": Accuracy(lambda x: Leaf("str", x),
                                     lambda x: x.value),
                "bleu": Bleu(lambda x: Leaf("str", x),
                             lambda x: x.value),
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
                "accuracy": Accuracy(lambda x: Leaf("str", x),
                                     lambda x: x.value),
                "bleu": Bleu(lambda x: Leaf("str", x),
                             lambda x: x.value),
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
