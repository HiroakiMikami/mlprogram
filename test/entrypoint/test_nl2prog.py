import unittest
import tempfile
import os
import json
import torch
from torch import nn
from torch import optim

from mlprogram.entrypoint.nl2prog import train, evaluate
from mlprogram.asts import Leaf
from mlprogram.utils.data import ListDataset
from mlprogram.synthesizers import Result
from mlprogram.metrics import Accuracy, Bleu


class TestTrain(unittest.TestCase):
    def prepare_dataset(self):
        return ListDataset([0, 1, 2])

    def prepare_model(self):
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.m = nn.Linear(1, 1)

            def forward(self, kwargs):
                kwargs["value"] = self.m(kwargs["value"])
                return kwargs

        return DummyModel()

    def prepare_optimizer(self, model):
        return optim.SGD(model.parameters(), 0.1)

    def collate(self, elems):
        B = len(elems)
        tensor = torch.tensor(elems).reshape(B, 1).float()
        return {"value": tensor, "target": tensor}

    def test_happy_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = os.path.join(tmpdir, "ws")
            output = os.path.join(tmpdir, "out")
            model = self.prepare_model()
            train(ws, output,
                  self.prepare_dataset(),
                  model,
                  self.prepare_optimizer(model),
                  lambda kwargs: nn.MSELoss()(kwargs["value"],
                                              kwargs["target"]),
                  lambda kwargs: nn.MSELoss()(kwargs["value"],
                                              kwargs["target"]),
                  self.collate, 1, 2)
            self.assertTrue(os.path.exists(
                os.path.join(ws, "snapshot_iter_6")))
            self.assertTrue(os.path.exists(os.path.join(ws, "log")))
            with open(os.path.join(ws, "log")) as file:
                log = json.load(file)
            self.assertTrue(isinstance(log, list))
            self.assertEqual(2, len(log))
            self.assertEqual(2, len(os.listdir(os.path.join(ws, "model"))))

            self.assertTrue(os.path.exists(os.path.join(output, "log.json")))
            with open(os.path.join(output, "log.json")) as file:
                log = json.load(file)
            self.assertTrue(isinstance(log, list))
            self.assertEqual(2, len(log))
            self.assertEqual(2, len(os.listdir(os.path.join(output, "model"))))

    def test_remove_old_snapshots(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = os.path.join(tmpdir, "ws")
            output = os.path.join(tmpdir, "out")
            model = self.prepare_model()
            train(ws, output,
                  self.prepare_dataset(),
                  model, self.prepare_optimizer(model),
                  lambda kwargs: nn.MSELoss()(kwargs["value"],
                                              kwargs["target"]),
                  lambda kwargs: nn.MSELoss()(kwargs["value"],
                                              kwargs["target"]),
                  self.collate, 1, 2,
                  num_checkpoints=1)
            self.assertTrue(os.path.exists(
                os.path.join(ws, "snapshot_iter_6")))

    def test_resume_from_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = os.path.join(tmpdir, "ws")
            output = os.path.join(tmpdir, "out")
            model = self.prepare_model()
            optimizer = self.prepare_optimizer(model)
            train(ws, output,
                  self.prepare_dataset(),
                  model, optimizer,
                  lambda kwargs: nn.MSELoss()(kwargs["value"],
                                              kwargs["target"]),
                  lambda kwargs: nn.MSELoss()(kwargs["value"],
                                              kwargs["target"]),
                  self.collate, 1, 1)
            with open(os.path.join(output, "log.json")) as file:
                log = json.load(file)

            train(ws, output,
                  self.prepare_dataset(),
                  model, optimizer,
                  lambda kwargs: nn.MSELoss()(kwargs["value"],
                                              kwargs["target"]),
                  lambda kwargs: nn.MSELoss()(kwargs["value"],
                                              kwargs["target"]),
                  self.collate, 1, 2)
            self.assertTrue(os.path.exists(
                os.path.join(ws, "snapshot_iter_6")))
            with open(os.path.join(output, "log.json")) as file:
                log2 = json.load(file)
            self.assertEqual(log[0], log2[0])
            self.assertEqual(2, len(log2))


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
