import unittest
import tempfile
import os
import json
import torch
from torch import nn
from torch import optim

from mlprogram.gin import workspace
from mlprogram.gin.nl2prog import train, evaluate
from mlprogram.asts import Leaf
from mlprogram.utils.data import ListDataset
from mlprogram.decoders import Result
from mlprogram.metrics import Accuracy, Bleu


class TestTrain(unittest.TestCase):
    def prepare_dataset(self, dataset_path):
        workspace.put(dataset_path, {"train": ListDataset([0, 1, 2])})

    def prepare_encoder(self):
        workspace.put("encoder", 0)

    def prepare_model(self, model_path):
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.m = nn.Linear(1, 1)

            def forward(self, **kwargs):
                kwargs["value"] = self.m(kwargs["value"])
                return kwargs

        workspace.put(model_path, DummyModel())

    def prepare_optimizer(self, optimizer_path):
        model = workspace.get("model")
        workspace.put(optimizer_path, optim.SGD(model.parameters(), 0.1))

    def collate_fn(self, elems):
        B = len(elems)
        tensor = torch.tensor(elems).reshape(B, 1).float()
        return {"value": tensor, "target": tensor}

    def test_happy_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = os.path.join(tmpdir, "ws")
            output = os.path.join(tmpdir, "out")
            train("dataset", "model", "optimizer", set(["encoder"]),
                  ws, output,
                  self.prepare_dataset, self.prepare_encoder,
                  self.prepare_model, self.prepare_optimizer,
                  lambda: lambda x: x,
                  lambda **kwargs: nn.MSELoss()(kwargs["value"],
                                                kwargs["target"]),
                  lambda **kwargs: nn.MSELoss()(kwargs["value"],
                                                kwargs["target"]),
                  self.collate_fn, 1, 2)
            self.assertTrue(os.path.exists(os.path.join(ws,
                                                        "encoder.pt")))
            self.assertEqual({"encoder": 0}, torch.load(
                os.path.join(ws, "encoder.pt")))
            self.assertTrue(os.path.exists(
                os.path.join(ws, "checkpoint", "0.pt")))
            self.assertTrue(os.path.exists(
                os.path.join(ws, "checkpoint", "1.pt")))
            self.assertTrue(os.path.exists(os.path.join(ws, "log.json")))
            with open(os.path.join(ws, "log.json")) as file:
                log = json.load(file)
            self.assertTrue(isinstance(log, list))
            self.assertEqual(2, len(log))
            self.assertEqual(2, len(os.listdir(os.path.join(ws, "model"))))

            self.assertTrue(os.path.exists(os.path.join(output,
                                                        "encoder.pt")))
            self.assertEqual({"encoder": 0}, torch.load(
                os.path.join(output, "encoder.pt")))
            self.assertTrue(os.path.exists(os.path.join(output, "log.json")))
            with open(os.path.join(ws, "log.json")) as file:
                log = json.load(file)
            self.assertTrue(isinstance(log, list))
            self.assertEqual(2, len(log))
            self.assertEqual(2, len(os.listdir(os.path.join(output, "model"))))

    def test_reuse_encoder(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            def dummpy_prepare_encoder():
                raise NotImplementedError

            ws = os.path.join(tmpdir, "ws")
            output = os.path.join(tmpdir, "out")
            os.makedirs(ws)
            torch.save({"encoder": 1}, os.path.join(ws, "encoder.pt"))
            train("dataset", "model", "optimizer", set(["encoder"]),
                  ws, output,
                  self.prepare_dataset, dummpy_prepare_encoder,
                  self.prepare_model, self.prepare_optimizer,
                  lambda: lambda x: x,
                  lambda **kwargs: nn.MSELoss()(kwargs["value"],
                                                kwargs["target"]),
                  lambda **kwargs: nn.MSELoss()(kwargs["value"],
                                                kwargs["target"]),
                  self.collate_fn, 1, 2)
            self.assertTrue(os.path.exists(os.path.join(ws,
                                                        "encoder.pt")))
            self.assertEqual({"encoder": 1}, torch.load(
                os.path.join(ws, "encoder.pt")))

    def test_remove_old_snapshots(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = os.path.join(tmpdir, "ws")
            output = os.path.join(tmpdir, "out")
            train("dataset", "model", "optimizer", set(["encoder"]),
                  ws, output,
                  self.prepare_dataset, self.prepare_encoder,
                  self.prepare_model, self.prepare_optimizer,
                  lambda: lambda x: x,
                  lambda **kwargs: nn.MSELoss()(kwargs["value"],
                                                kwargs["target"]),
                  lambda **kwargs: nn.MSELoss()(kwargs["value"],
                                                kwargs["target"]),
                  self.collate_fn, 1, 2,
                  num_checkpoints=1)
            self.assertFalse(os.path.exists(
                os.path.join(ws, "checkpoint", "0.pt")))
            self.assertTrue(os.path.exists(
                os.path.join(ws, "checkpoint", "1.pt")))

    def test_resume_from_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = os.path.join(tmpdir, "ws")
            output = os.path.join(tmpdir, "out")
            train("dataset", "model", "optimizer", set(["encoder"]),
                  ws, output,
                  self.prepare_dataset, self.prepare_encoder,
                  self.prepare_model, self.prepare_optimizer,
                  lambda: lambda x: x,
                  lambda **kwargs: nn.MSELoss()(kwargs["value"],
                                                kwargs["target"]),
                  lambda **kwargs: nn.MSELoss()(kwargs["value"],
                                                kwargs["target"]),
                  self.collate_fn, 1, 1)
            with open(os.path.join(output, "log.json")) as file:
                log = json.load(file)

            train("dataset", "model", "optimizer", set(["encoder"]),
                  ws, output,
                  self.prepare_dataset, self.prepare_encoder,
                  self.prepare_model, self.prepare_optimizer,
                  lambda: lambda x: x,
                  lambda **kwargs: nn.MSELoss()(kwargs["value"],
                                                kwargs["target"]),
                  lambda **kwargs: nn.MSELoss()(kwargs["value"],
                                                kwargs["target"]),
                  self.collate_fn, 1, 2)
            self.assertTrue(os.path.exists(
                os.path.join(ws, "checkpoint", "0.pt")))
            self.assertTrue(os.path.exists(
                os.path.join(ws, "checkpoint", "1.pt")))
            with open(os.path.join(output, "log.json")) as file:
                log2 = json.load(file)
            self.assertEqual(log[0], log2[0])
            self.assertEqual(2, len(log2))


class TestEvaluate(unittest.TestCase):
    def prepare_dataset(self, dataset_path):
        workspace.put(dataset_path,
                      {"test": ListDataset([{"input": ["query"],
                                             "ground_truth": ["name0"]}]),
                       "valid": ListDataset([{"input": ["query"],
                                              "ground_truth": ["name0"]}])})

    def prepare_synthesizer(self, synthesizer_path):
        class MockSynthesizer:
            def load_state_dict(self, state_dict):
                self.state_dict = state_dict

            def __call__(self, query):
                yield Result(Leaf("str", self.state_dict["name"]),
                             self.state_dict["score"])
        workspace.put(synthesizer_path, MockSynthesizer())

    def test_happy_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input = os.path.join(tmpdir, "input")
            ws = os.path.join(tmpdir, "workspace")
            output = os.path.join(tmpdir, "output")
            os.makedirs(input)
            os.makedirs(os.path.join(input, "model"))
            torch.save({"encoder": 0}, os.path.join(input, "encoder.pt"))
            torch.save({"model": {"score": 0.5, "name": "tmp"}},
                       os.path.join(input, "model", "0"))
            torch.save({"model": {"score": 1.0, "name": "tmp"}},
                       os.path.join(input, "model", "1"))
            evaluate("dataset", "synthesizer", set(["encoder"]),
                     input, ws, output,
                     self.prepare_dataset, self.prepare_synthesizer,
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
            torch.save({"encoder": 0}, os.path.join(input, "encoder.pt"))
            torch.save({"model": {"score": 0.5, "name": "tmp"}},
                       os.path.join(input, "model", "0"))
            evaluate("dataset", "synthesizer", set(["encoder"]),
                     input, ws, output,
                     self.prepare_dataset, self.prepare_synthesizer,
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
            evaluate("dataset", "synthesizer", set(["encoder"]),
                     input, ws, output,
                     self.prepare_dataset, self.prepare_synthesizer,
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
