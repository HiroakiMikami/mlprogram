import json
import multiprocessing as mp
import os
import tempfile

import numpy as np
import pytest
import torch
from pytorch_pfn_extras.reporting import report
from torch import nn, optim

from mlprogram import distributed
from mlprogram.builtins import Environment
from mlprogram.entrypoint import train_REINFORCE, train_supervised
from mlprogram.entrypoint.train import Epoch, Iteration
from mlprogram.synthesizers import Result
from mlprogram.utils.data import Collate, CollateOptions, ListDataset

context = mp.get_context("spawn")


class MockSynthesizer:
    def __init__(self, model):
        self.model = model

    def __call__(self, input, n_required_output=None):
        n_required_output = n_required_output or 1
        for _ in range(n_required_output):
            yield Result(input["value"], 0, True, 1)


@pytest.fixture
def synthesizer(model):
    return MockSynthesizer(model)


def reward(sample, output):
    return sample["value"] == output


collate = Collate(output=CollateOptions(False, 0, 0),
                  value=CollateOptions(False, 0, 0),
                  reward=CollateOptions(False, 0, 0),
                  ground_truth=CollateOptions(False, 0, 0))


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = nn.Linear(1, 1)

    def forward(self, kwargs):
        kwargs["value"] = self.m(kwargs["value"].float())
        return kwargs


@pytest.fixture
def model():
    return DummyModel()


class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, input):
        return self.loss(
            input["value"].float(),
            input["ground_truth"].float()
        )


@pytest.fixture
def loss_fn():
    return Loss()


@pytest.fixture
def optimizer(model):
    return optim.SGD(model.parameters(), 0.1)


class MockEvaluate(object):
    def __init__(self, key):
        self.key = key

    def __call__(self):
        report({self.key: 0.0})


@pytest.fixture
def dataset():
    return ListDataset([
        Environment(
            {"value": torch.tensor(0), "ground_truth": torch.tensor(0)},
            set(["ground_truth"]),
        ),
        Environment(
            {"value": torch.tensor(1), "ground_truth": torch.tensor(1)},
            set(["ground_truth"]),
        ),
        Environment(
            {"value": torch.tensor(2), "ground_truth": torch.tensor(2)},
            set(["ground_truth"]),
        ),
    ])


@pytest.fixture
def iterable_dataset():
    class MockDataset(torch.utils.data.IterableDataset):
        def __iter__(self):
            class InternalIterator:
                def __init__(self):
                    self.n = 0

                def __next__(self) -> Environment:
                    self.n += 1
                    return Environment(
                        {"value": torch.tensor(self.n),
                         "ground_truth": torch.tensor(self.n)},
                        set(["ground_truth"])
                    )
            return InternalIterator()
    return MockDataset()


def _run(init_dir, dataset, model, loss_fn, optimizer, rank):
    with tempfile.TemporaryDirectory() as tmpdir:
        distributed.initialize(init_dir, rank, 2)

        ws = os.path.join(tmpdir, "ws")
        output = os.path.join(tmpdir, "out")
        train_supervised(ws, output,
                         dataset,
                         model,
                         optimizer,
                         loss_fn,
                         MockEvaluate("key"), "key",
                         collate.collate, 1, Epoch(2),
                         n_dataloader_worker=0)
        if rank == 0:
            assert os.path.exists(os.path.join(ws, "snapshot_iter_2"))

            assert os.path.exists(os.path.join(output, "log.json"))
            with open(os.path.join(output, "log.json")) as file:
                log = json.load(file)
            assert isinstance(log, list)
            assert 1 == len(log)
            assert 1 == len(os.listdir(os.path.join(output, "model")))
            assert os.path.exists(os.path.join(output, "model.pt"))
            assert os.path.exists(os.path.join(output, "optimizer.pt"))
        else:
            assert not os.path.exists(os.path.join(ws, "snapshot_iter_2"))
            assert not os.path.exists(os.path.join(output, "log.json"))
            assert not os.path.exists(os.path.join(output, "model"))
        return model.state_dict(), optimizer.state_dict()


class TestTrainSupervised(object):
    def test_happy_path(self, dataset, model, loss_fn, optimizer):
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = os.path.join(tmpdir, "ws")
            output = os.path.join(tmpdir, "out")
            train_supervised(ws, output,
                             dataset,
                             model,
                             optimizer,
                             loss_fn,
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

    def test_multiprocess(self, dataset, model, loss_fn, optimizer):
        with tempfile.TemporaryDirectory() as init_dir:
            with context.Pool(2) as pool:
                procs = []
                for i in range(2):
                    p = pool.apply_async(
                        _run,
                        args=(init_dir, dataset, model, loss_fn, optimizer, i),
                    )
                    procs.append(p)
                out = [p.get() for p in procs]

        m0 = out[0][0]
        m1 = out[1][0]
        assert m0.keys() == m1.keys()
        for key in m0.keys():
            assert np.array_equal(m0[key], m1[key])

    def test_resume_from_eval_mode(self, dataset, loss_fn):
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.m = nn.Linear(1, 1)

            def forward(self, kwargs):
                assert self.training
                kwargs["value"] = self.m(kwargs["value"].float())
                return kwargs

        class MockEvaluate(object):
            def __init__(self, key, model):
                self.key = key
                self.model = model

            def __call__(self):
                self.model.eval()
                report({self.key: 0.0})

        with tempfile.TemporaryDirectory() as tmpdir:
            ws = os.path.join(tmpdir, "ws")
            output = os.path.join(tmpdir, "out")
            model = DummyModel()
            train_supervised(ws, output,
                             dataset,
                             model,
                             torch.optim.SGD(model.parameters(), lr=0.1),
                             loss_fn,
                             MockEvaluate("key", model), "key",
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

    def test_threshold(self, dataset, model, loss_fn, optimizer):
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = os.path.join(tmpdir, "ws")
            output = os.path.join(tmpdir, "out")
            train_supervised(ws, output,
                             dataset,
                             model,
                             optimizer,
                             loss_fn,
                             MockEvaluate("key"), "key",
                             collate.collate, 1, Epoch(2),
                             threshold=0.0)
            assert 1 == len(os.listdir(os.path.join(ws, "model")))

            assert 1 == len(os.listdir(os.path.join(output, "model")))
            assert os.path.exists(os.path.join(output, "model.pt"))
            assert os.path.exists(os.path.join(output, "optimizer.pt"))

    def test_skip_evaluation(self, dataset, model, loss_fn, optimizer):
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = os.path.join(tmpdir, "ws")
            output = os.path.join(tmpdir, "out")
            train_supervised(ws, output,
                             dataset,
                             model,
                             optimizer,
                             loss_fn,
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

    def test_remove_old_snapshots(self, dataset, model, loss_fn, optimizer):
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = os.path.join(tmpdir, "ws")
            output = os.path.join(tmpdir, "out")
            train_supervised(ws, output,
                             dataset,
                             model, optimizer,
                             loss_fn,
                             MockEvaluate("key"), "key",
                             collate.collate, 1, Epoch(2))
            assert os.path.exists(
                os.path.join(ws, "snapshot_iter_6"))

    def test_resume_from_checkpoint(self, dataset, model, loss_fn, optimizer):
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = os.path.join(tmpdir, "ws")
            output = os.path.join(tmpdir, "out")
            train_supervised(ws, output,
                             dataset,
                             model, optimizer,
                             loss_fn,
                             MockEvaluate("key"), "key",
                             collate.collate, 1, Epoch(1))
            with open(os.path.join(output, "log.json")) as file:
                log = json.load(file)

            train_supervised(ws, output,
                             dataset,
                             model, optimizer,
                             loss_fn,
                             MockEvaluate("key"), "key",
                             collate.collate, 1, Epoch(2))
            assert os.path.exists(
                os.path.join(ws, "snapshot_iter_6"))
            with open(os.path.join(output, "log.json")) as file:
                log2 = json.load(file)
            assert log[0] == log2[0]
            assert 2 == len(log2)

    def test_finish_by_iteration(self, dataset, model, loss_fn, optimizer):
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = os.path.join(tmpdir, "ws")
            output = os.path.join(tmpdir, "out")
            train_supervised(ws, output,
                             dataset,
                             model,
                             optimizer,
                             loss_fn,
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

    def test_iterable_dataset(self, iterable_dataset, model, loss_fn, optimizer):
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = os.path.join(tmpdir, "ws")
            output = os.path.join(tmpdir, "out")
            train_supervised(ws, output,
                             iterable_dataset,
                             model,
                             optimizer,
                             loss_fn,
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
    def loss_fn(self, input):
        return nn.MSELoss()(input["value"].float(),
                            input["ground_truth"].float())

    def test_happy_path(self, dataset, model, loss_fn, optimizer, synthesizer):
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = os.path.join(tmpdir, "ws")
            output = os.path.join(tmpdir, "out")
            train_REINFORCE(output, ws, output,
                            dataset,
                            synthesizer,
                            model,
                            optimizer,
                            lambda x: loss_fn(x) * x["reward"],
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

    def test_pretrained_model(self, dataset, model, loss_fn, optimizer, synthesizer):
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = os.path.join(tmpdir, "ws")
            input = os.path.join(tmpdir, "in")
            output = os.path.join(tmpdir, "out")
            model2 = DummyModel()
            model2.m.bias[:] = np.nan
            os.makedirs(input)
            torch.save(model2.state_dict(), os.path.join(input, "model.pt"))

            train_REINFORCE(input, ws, output,
                            dataset,
                            synthesizer,
                            model,
                            optimizer,
                            lambda x: loss_fn(x) * x["reward"],
                            MockEvaluate("key"), "key",
                            reward,
                            collate.collate,
                            1, 1, Epoch(2),
                            use_pretrained_model=True)
            assert not os.path.exists(os.path.join(ws, "log"))

    def test_pretrained_optimizer(
        self, dataset, model, loss_fn, optimizer, synthesizer
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = os.path.join(tmpdir, "ws")
            input = os.path.join(tmpdir, "in")
            output = os.path.join(tmpdir, "out")
            optimizer = torch.optim.SGD(model.parameters(), lr=np.nan)
            os.makedirs(input)
            torch.save(optimizer.state_dict(),
                       os.path.join(input, "optimizer.pt"))

            train_REINFORCE(input, ws, output,
                            dataset,
                            synthesizer,
                            model,
                            optimizer,
                            lambda x: loss_fn(x) * x["reward"],
                            MockEvaluate("key"), "key",
                            reward,
                            collate.collate,
                            1, 1, Epoch(2),
                            use_pretrained_optimizer=True)
            assert not os.path.exists(os.path.join(ws, "log"))
