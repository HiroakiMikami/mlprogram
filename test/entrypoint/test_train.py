import json
import os
import tempfile

import numpy as np
import torch
from pytorch_pfn_extras.reporting import report
from torch import nn, optim

from mlprogram import Environment
from mlprogram.entrypoint import train_REINFORCE, train_supervised
from mlprogram.entrypoint.train import Epoch, Iteration
from mlprogram.synthesizers import Result
from mlprogram.utils.data import Collate, CollateOptions, ListDataset


class MockSynthesizer:
    def __init__(self, model):
        self.model = model

    def __call__(self, input, n_required_output=None):
        n_required_output = n_required_output or 1
        for _ in range(n_required_output):
            yield Result(input.inputs["value"], 0, True, 1)


def reward(sample, output):
    return sample.inputs["value"] == output


collate = Collate(device=torch.device("cpu"),
                  output=CollateOptions(False, 0, 0),
                  value=CollateOptions(False, 0, 0),
                  reward=CollateOptions(False, 0, 0),
                  ground_truth=CollateOptions(False, 0, 0))


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = nn.Linear(1, 1)

    def forward(self, kwargs):
        kwargs.outputs["value"] = self.m(kwargs.inputs["value"].float())
        return kwargs


class MockEvaluate(object):
    def __init__(self, key):
        self.key = key

    def __call__(self):
        report({self.key: 0.0})


class TestTrainSupervised(object):
    def prepare_dataset(self):
        return ListDataset([
            Environment(inputs={"value": torch.tensor(0)},
                        supervisions={"value": torch.tensor(0)}),
            Environment(inputs={"value": torch.tensor(1)},
                        supervisions={"value": torch.tensor(1)}),
            Environment(inputs={"value": torch.tensor(2)},
                        supervisions={"value": torch.tensor(2)})
        ])

    def prepare_iterable_dataset(self):
        class MockDataset(torch.utils.data.IterableDataset):
            def __iter__(self):
                class InternalIterator:
                    def __init__(self):
                        self.n = 0

                    def __next__(self) -> Environment:
                        self.n += 1
                        return Environment(
                            inputs={"value": torch.tensor(self.n)},
                            supervisions={"value": torch.tensor(self.n)}
                        )
                return InternalIterator()
        return MockDataset()

    def prepare_model(self):
        return DummyModel()

    def prepare_optimizer(self, model):
        return optim.SGD(model.parameters(), 0.1)

    def loss_fn(self, input):
        return nn.MSELoss()(input.outputs["value"].float(),
                            input.supervisions["value"].float())

    def test_happy_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = os.path.join(tmpdir, "ws")
            output = os.path.join(tmpdir, "out")
            model = self.prepare_model()
            train_supervised(ws, output,
                             self.prepare_dataset(),
                             model,
                             self.prepare_optimizer(model),
                             self.loss_fn,
                             MockEvaluate("key"), "key",
                             collate.collate, 1, Epoch(2))
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
                             self.loss_fn,
                             MockEvaluate("key"), "key",
                             collate.collate, 1, Epoch(2),
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
                             self.loss_fn,
                             None, "key",
                             collate.collate, 1, Epoch(2))
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
                             self.loss_fn,
                             MockEvaluate("key"), "key",
                             collate.collate, 1, Epoch(2))
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
                             self.loss_fn,
                             MockEvaluate("key"), "key",
                             collate.collate, 1, Epoch(1))
            with open(os.path.join(output, "log.json")) as file:
                log = json.load(file)

            train_supervised(ws, output,
                             self.prepare_dataset(),
                             model, optimizer,
                             self.loss_fn,
                             MockEvaluate("key"), "key",
                             collate.collate, 1, Epoch(2))
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
                             self.loss_fn,
                             MockEvaluate("key"), "key",
                             collate.collate, 1, Iteration(2),
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
                             self.loss_fn,
                             MockEvaluate("key"), "key",
                             collate.collate, 1, Iteration(2),
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
            Environment(inputs={"value": torch.tensor(0)}),
            Environment(inputs={"value": torch.tensor(1)}),
            Environment(inputs={"value": torch.tensor(2)})
        ])

    def prepare_model(self):
        return DummyModel()

    def prepare_optimizer(self, model):
        return optim.SGD(model.parameters(), 0.1)

    def prepare_synthesizer(self, model):
        return MockSynthesizer(model)

    def loss_fn(self, input):
        return nn.MSELoss()(input.outputs["value"].float(),
                            input.supervisions["ground_truth"].float())

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
                            lambda x: self.loss_fn(x) * x.inputs["reward"],
                            MockEvaluate("key"), "key",
                            reward,
                            collate.collate,
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
                            lambda x: self.loss_fn(x) * x.inputs["reward"],
                            MockEvaluate("key"), "key",
                            reward,
                            collate.collate,
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
                            lambda x: self.loss_fn(x) * x.inputs["reward"],
                            MockEvaluate("key"), "key",
                            reward,
                            collate.collate,
                            1, 1, Epoch(2),
                            use_pretrained_optimizer=True)
            assert not os.path.exists(os.path.join(ws, "log"))
