import numpy as np
import tempfile
import os
import json
import torch
from torch import nn
from torch import optim

from pytorch_pfn_extras.reporting import report
from mlprogram.entrypoint.train import Epoch, Iteration
from mlprogram.entrypoint import train_supervised, train_REINFORCE
from mlprogram.utils.data import ListDataset
from mlprogram.synthesizers import Result


class MockSynthesizer:
    def __init__(self, model):
        self.model = model

    def __call__(self, input, n_required_output=None):
        n_required_output = n_required_output or 1
        for _ in range(n_required_output):
            yield Result({"output": input["value"]}, 0, True, 1)


def reward(sample, output):
    return sample["value"] == output["output"]


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = nn.Linear(1, 1)

    def forward(self, kwargs):
        kwargs["value"] = self.m(kwargs["value"])
        return kwargs


class MockEvaluate(object):
    def __init__(self, key):
        self.key = key

    def __call__(self):
        report({self.key: 0.0})


class TestTrainSupervised(object):
    def prepare_dataset(self):
        return ListDataset([0, 1, 2])

    def prepare_iterable_dataset(self):
        class MockDataset(torch.utils.data.IterableDataset):
            def __iter__(self):
                class InternalIterator:
                    def __init__(self):
                        self.n = 0

                    def __next__(self) -> int:
                        self.n += 1
                        return self.n
                return InternalIterator()
        return MockDataset()

    def prepare_model(self):
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
                             MockEvaluate("key"), "key",
                             self.collate, 1, Epoch(2))
            assert os.path.exists(
                os.path.join(ws, "snapshot_iter_6"))
            assert os.path.exists(os.path.join(ws, "log"))
            with open(os.path.join(ws, "log")) as file:
                log = json.load(file)
            assert isinstance(log, list)
            assert 1 == len(log)
            assert 1 == len(os.listdir(os.path.join(ws, "model")))

            assert os.path.exists(os.path.join(output, "log.json"))
            with open(os.path.join(output, "log.json")) as file:
                log = json.load(file)
            assert isinstance(log, list)
            assert 1 == len(log)
            assert 1 == len(os.listdir(os.path.join(output, "model")))
            assert os.path.exists(os.path.join(output, "model.pt"))
            assert os.path.exists(os.path.join(output, "optimizer.pt"))

    def test_threshold(self):
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
                             MockEvaluate("key"), "key",
                             self.collate, 1, Epoch(2),
                             threshold=0.0)
            assert 1 == len(os.listdir(os.path.join(ws, "model")))

            assert 1 == len(os.listdir(os.path.join(output, "model")))
            assert os.path.exists(os.path.join(output, "model.pt"))
            assert os.path.exists(os.path.join(output, "optimizer.pt"))

    def test_skip_evaluation(self):
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
                             None, "key",
                             self.collate, 1, Epoch(2))
            assert os.path.exists(
                os.path.join(ws, "snapshot_iter_6"))
            assert os.path.exists(os.path.join(ws, "log"))
            with open(os.path.join(ws, "log")) as file:
                log = json.load(file)
            assert isinstance(log, list)
            assert 1 == len(log)
            assert 0 == len(os.listdir(os.path.join(ws, "model")))

            assert os.path.exists(os.path.join(output, "log.json"))
            with open(os.path.join(output, "log.json")) as file:
                log = json.load(file)
            assert isinstance(log, list)
            assert 1 == len(log)
            assert 0 == len(os.listdir(os.path.join(output, "model")))
            assert os.path.exists(os.path.join(output, "model.pt"))
            assert os.path.exists(os.path.join(output, "optimizer.pt"))

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
                             MockEvaluate("key"), "key",
                             self.collate, 1, Epoch(2))
            assert os.path.exists(
                os.path.join(ws, "snapshot_iter_6"))

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
                             MockEvaluate("key"), "key",
                             self.collate, 1, Epoch(1))
            with open(os.path.join(output, "log.json")) as file:
                log = json.load(file)

            train_supervised(ws, output,
                             self.prepare_dataset(),
                             model, optimizer,
                             lambda kwargs: nn.MSELoss()(kwargs["value"],
                                                         kwargs["target"]),
                             MockEvaluate("key"), "key",
                             self.collate, 1, Epoch(2))
            assert os.path.exists(
                os.path.join(ws, "snapshot_iter_6"))
            with open(os.path.join(output, "log.json")) as file:
                log2 = json.load(file)
            assert log[0] == log2[0]
            assert 2 == len(log2)

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
                             MockEvaluate("key"), "key",
                             self.collate, 1, Iteration(2),
                             evaluation_interval=Iteration(1))
            assert os.path.exists(
                os.path.join(ws, "snapshot_iter_2"))
            assert os.path.exists(os.path.join(ws, "log"))
            with open(os.path.join(ws, "log")) as file:
                log = json.load(file)
            assert isinstance(log, list)
            assert 1 == len(log)
            assert 1 == len(os.listdir(os.path.join(ws, "model")))

            assert os.path.exists(os.path.join(output, "log.json"))
            with open(os.path.join(output, "log.json")) as file:
                log = json.load(file)
            assert isinstance(log, list)
            assert 1 == len(log)
            assert 1 == len(os.listdir(os.path.join(output, "model")))

    def test_iterable_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = os.path.join(tmpdir, "ws")
            output = os.path.join(tmpdir, "out")
            model = self.prepare_model()
            train_supervised(ws, output,
                             self.prepare_iterable_dataset(),
                             model,
                             self.prepare_optimizer(model),
                             lambda kwargs: nn.MSELoss()(kwargs["value"],
                                                         kwargs["target"]),
                             MockEvaluate("key"), "key",
                             self.collate, 1, Iteration(2),
                             evaluation_interval=Iteration(1))
            assert os.path.exists(
                os.path.join(ws, "snapshot_iter_2"))
            assert os.path.exists(os.path.join(ws, "log"))
            with open(os.path.join(ws, "log")) as file:
                log = json.load(file)
            assert isinstance(log, list)
            assert 1 == len(log)
            assert 1 == len(os.listdir(os.path.join(ws, "model")))

            assert os.path.exists(os.path.join(output, "log.json"))
            with open(os.path.join(output, "log.json")) as file:
                log = json.load(file)
            assert isinstance(log, list)
            assert 1 == len(log)
            assert 1 == len(os.listdir(os.path.join(output, "model")))


class TestTrainREINFORCE(object):
    def prepare_dataset(self):
        return ListDataset([
            {"value": 0},
            {"value": 1},
            {"value": 2}
        ])

    def prepare_model(self):
        return DummyModel()

    def prepare_optimizer(self, model):
        return optim.SGD(model.parameters(), 0.1)

    def collate(self, elems):
        B = len(elems)
        output = [elem["ground_truth"]["output"] for elem in elems]
        tensor = torch.tensor(output).reshape(B, 1).float()
        if "reward" in elems[0]:
            reward = torch.stack([elem["reward"] for elem in elems])
            return {"value": tensor, "target": tensor, "reward": reward}
        else:
            return {"value": tensor, "target": tensor}

    def prepare_synthesizer(self, model):
        return MockSynthesizer(model)

    def test_happy_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = os.path.join(tmpdir, "ws")
            output = os.path.join(tmpdir, "out")
            model = self.prepare_model()
            train_REINFORCE(output, ws, output,
                            self.prepare_dataset(),
                            self.prepare_synthesizer(model),
                            model,
                            self.prepare_optimizer(model),
                            lambda kwargs: nn.MSELoss()(kwargs["value"],
                                                        kwargs["target"]
                                                        ) * kwargs["reward"],
                            MockEvaluate("key"), "key",
                            reward,
                            self.collate,
                            1, 1, Epoch(2))
            assert os.path.exists(
                os.path.join(ws, "snapshot_iter_6"))
            assert os.path.exists(os.path.join(ws, "log"))
            with open(os.path.join(ws, "log")) as file:
                log = json.load(file)
            assert isinstance(log, list)
            assert 1 == len(log)
            assert 1 == len(os.listdir(os.path.join(ws, "model")))

            assert os.path.exists(os.path.join(output, "log.json"))
            with open(os.path.join(output, "log.json")) as file:
                log = json.load(file)
            assert isinstance(log, list)
            assert 1 == len(log)
            assert 1 == len(os.listdir(os.path.join(output, "model")))
            assert os.path.exists(os.path.join(output, "model.pt"))
            assert os.path.exists(
                os.path.join(output, "optimizer.pt"))

    def test_pretrained_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = os.path.join(tmpdir, "ws")
            input = os.path.join(tmpdir, "in")
            output = os.path.join(tmpdir, "out")
            model = self.prepare_model()
            model2 = self.prepare_model()
            model2.m.bias[:] = np.nan
            os.makedirs(input)
            torch.save(model2.state_dict(), os.path.join(input, "model.pt"))

            train_REINFORCE(input, ws, output,
                            self.prepare_dataset(),
                            self.prepare_synthesizer(model),
                            model,
                            self.prepare_optimizer(model),
                            lambda kwargs: nn.MSELoss()(kwargs["value"],
                                                        kwargs["target"]
                                                        ) * kwargs["reward"],
                            MockEvaluate("key"), "key",
                            reward,
                            self.collate,
                            1, 1, Epoch(2),
                            use_pretrained_model=True)
            assert not os.path.exists(os.path.join(ws, "log"))

    def test_pretrained_optimizer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = os.path.join(tmpdir, "ws")
            input = os.path.join(tmpdir, "in")
            output = os.path.join(tmpdir, "out")
            model = self.prepare_model()
            optimizer = torch.optim.SGD(model.parameters(), lr=np.nan)
            os.makedirs(input)
            torch.save(optimizer.state_dict(),
                       os.path.join(input, "optimizer.pt"))

            train_REINFORCE(input, ws, output,
                            self.prepare_dataset(),
                            self.prepare_synthesizer(model),
                            model,
                            self.prepare_optimizer(model),
                            lambda kwargs: nn.MSELoss()(kwargs["value"],
                                                        kwargs["target"]
                                                        ) * kwargs["reward"],
                            MockEvaluate("key"), "key",
                            reward,
                            self.collate,
                            1, 1, Epoch(2),
                            use_pretrained_optimizer=True)
            assert not os.path.exists(os.path.join(ws, "log"))
