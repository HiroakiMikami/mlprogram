import unittest
import tempfile
import os
import json
import torch
from torch import nn
from torch import optim

from mlprogram.entrypoint.train import Epoch, Iteration
from mlprogram.entrypoint import train_supervised
from mlprogram.utils.data import ListDataset


class TestTrainSupervised(unittest.TestCase):
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
            train_supervised(ws, output,
                             self.prepare_dataset(),
                             model,
                             self.prepare_optimizer(model),
                             lambda kwargs: nn.MSELoss()(kwargs["value"],
                                                         kwargs["target"]),
                             lambda kwargs: nn.MSELoss()(kwargs["value"],
                                                         kwargs["target"]),
                             self.collate, 1, Epoch(2))
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
            train_supervised(ws, output,
                             self.prepare_dataset(),
                             model, self.prepare_optimizer(model),
                             lambda kwargs: nn.MSELoss()(kwargs["value"],
                                                         kwargs["target"]),
                             lambda kwargs: nn.MSELoss()(kwargs["value"],
                                                         kwargs["target"]),
                             self.collate, 1, Epoch(2),
                             num_checkpoints=1)
            self.assertTrue(os.path.exists(
                os.path.join(ws, "snapshot_iter_6")))

    def test_resume_from_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = os.path.join(tmpdir, "ws")
            output = os.path.join(tmpdir, "out")
            model = self.prepare_model()
            optimizer = self.prepare_optimizer(model)
            train_supervised(ws, output,
                             self.prepare_dataset(),
                             model, optimizer,
                             lambda kwargs: nn.MSELoss()(kwargs["value"],
                                                         kwargs["target"]),
                             lambda kwargs: nn.MSELoss()(kwargs["value"],
                                                         kwargs["target"]),
                             self.collate, 1, Epoch(1))
            with open(os.path.join(output, "log.json")) as file:
                log = json.load(file)

            train_supervised(ws, output,
                             self.prepare_dataset(),
                             model, optimizer,
                             lambda kwargs: nn.MSELoss()(kwargs["value"],
                                                         kwargs["target"]),
                             lambda kwargs: nn.MSELoss()(kwargs["value"],
                                                         kwargs["target"]),
                             self.collate, 1, Epoch(2))
            self.assertTrue(os.path.exists(
                os.path.join(ws, "snapshot_iter_6")))
            with open(os.path.join(output, "log.json")) as file:
                log2 = json.load(file)
            self.assertEqual(log[0], log2[0])
            self.assertEqual(2, len(log2))

    def test_finish_by_iteration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = os.path.join(tmpdir, "ws")
            output = os.path.join(tmpdir, "out")
            model = self.prepare_model()
            train_supervised(ws, output,
                             self.prepare_dataset(),
                             model,
                             self.prepare_optimizer(model),
                             lambda kwargs: nn.MSELoss()(kwargs["value"],
                                                         kwargs["target"]),
                             lambda kwargs: nn.MSELoss()(kwargs["value"],
                                                         kwargs["target"]),
                             self.collate, 1, Iteration(2),
                             interval=Iteration(1))
            self.assertTrue(os.path.exists(
                os.path.join(ws, "snapshot_iter_2")))
            self.assertTrue(os.path.exists(os.path.join(ws, "log")))
            with open(os.path.join(ws, "log")) as file:
                log = json.load(file)
            self.assertTrue(isinstance(log, list))
            self.assertEqual(1, len(log))
            self.assertEqual(2, len(os.listdir(os.path.join(ws, "model"))))

            self.assertTrue(os.path.exists(os.path.join(output, "log.json")))
            with open(os.path.join(output, "log.json")) as file:
                log = json.load(file)
            self.assertTrue(isinstance(log, list))
            self.assertEqual(1, len(log))
            self.assertEqual(2, len(os.listdir(os.path.join(output, "model"))))


if __name__ == "__main__":
    unittest.main()