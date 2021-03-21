import multiprocessing as mp
import os
import tempfile

import pytest
import torch

from mlprogram import distributed
from mlprogram.builtins import Environment
from mlprogram.entrypoint import evaluate
from mlprogram.entrypoint.evaluate import EvaluateSynthesizer, Result
from mlprogram.metrics import Accuracy, Bleu, use_environment
from mlprogram.synthesizers import Result as DecoderResult
from mlprogram.utils.data import ListDataset

context = mp.get_context("spawn")


class MockModel:
    def load_state_dict(self, state_dict):
        self.state_dict = state_dict

    def state_dict(self):
        return {}

    def to(self, *args, **kwargs):
        pass


class MockSynthesizer:
    def __init__(self, model):
        self.model = model

    def __call__(self, input):
        yield DecoderResult(self.model.state_dict["name"],
                            self.model.state_dict["score"],
                            True,
                            1)


def synthesize(input):
    input = input["query"]
    output = []
    if input == "query0":
        output = ["c0", "c1", "c2"]
    elif input == "query1":
        output = ["c2", "c3", "c0"]
    else:
        output = ["c2", "c3", "c5"]

    for i, s in enumerate(output):
        yield DecoderResult(s, -i, True, 1)


class TestEvaluateSynthesizer(object):
    def test_simple_case(self):
        accuracy = use_environment(
            Accuracy(), in_keys=["actual", ["ground_truth", "expected"]],
            value_key="actual"
        )
        dataset = ListDataset([
            Environment(
                {"query": "query0", "ground_truth": "c0"},
                set(["ground_truth"])
            ),
            Environment(
                {"query": "query1", "ground_truth": "c0"},
                set(["ground_truth"])
            ),
            Environment(
                {"query": "query2", "ground_truth": "c0"},
                set(["ground_truth"])
            ),
        ])
        results = EvaluateSynthesizer(dataset, synthesize,
                                      metrics={"accuracy": accuracy})()

        assert results.metrics == \
            {1: {"accuracy": 1.0 / 3.0}, 3: {"accuracy": 2.0 / 3.0}}
        assert 3 == len(results.results)
        results.results[0].time = 0.0
        results.results[1].time = 0.0
        results.results[2].time = 0.0
        assert Result({"query": "query0",
                       "ground_truth": "c0"},
                      ["c0", "c1", "c2"],
                      {1: {"accuracy": 1.0}, 3: {"accuracy": 1.0}},
                      True, 0.0) == results.results[0]
        assert Result({"query": "query1",
                       "ground_truth": "c0"},
                      ["c2", "c3", "c0"],
                      {1: {"accuracy": 0.0}, 3: {"accuracy": 1.0}},
                      True, 0.0) == results.results[1]
        assert Result({"query": "query2",
                       "ground_truth": "c0"},
                      ["c2", "c3", "c5"],
                      {1: {"accuracy": 0.0}, 3: {"accuracy": 0.0}},
                      True, 0.0) == results.results[2]

    def _run(self, init_dir, dataset, metrics, rank):
        distributed.initialize(init_dir, rank, 2)
        return EvaluateSynthesizer(dataset, synthesize,
                                   metrics=metrics)()

    def test_multiprocess(self):
        accuracy = use_environment(
            Accuracy(), in_keys=["actual", ["ground_truth", "expected"]],
            value_key="actual"
        )
        dataset = ListDataset([
            Environment(
                {"query": "query0", "ground_truth": "c0"},
                set(["ground_truth"])
            ),
            Environment(
                {"query": "query1", "ground_truth": "c0"},
                set(["ground_truth"])
            ),
            Environment(
                {"query": "query2", "ground_truth": "c0"},
                set(["ground_truth"])
            ),
        ])

        with tempfile.TemporaryDirectory() as init_dir:
            with context.Pool(2) as pool:
                procs = []
                for i in range(2):
                    p = pool.apply_async(
                        self._run,
                        args=(init_dir, dataset, {"accuracy": accuracy}, i),
                    )
                    procs.append(p)
                out = [p.get() for p in procs]
        r0 = out[0]
        r1 = out[1]

        assert r0 == r1

        results = r0
        assert results.metrics == {1: {"accuracy": 1.0 / 3},
                                   3: {"accuracy": 2.0 / 3}}
        assert 3 == len(results.results)
        results.results[0].time = 0.0
        results.results[1].time = 0.0
        results.results[2].time = 0.0
        results.results.sort(key=lambda x: x.sample["query"])
        assert Result({"query": "query0",
                       "ground_truth": "c0"},
                      ["c0", "c1", "c2"],
                      {1: {"accuracy": 1.0}, 3: {"accuracy": 1.0}},
                      True, 0.0) == results.results[0]
        assert Result({"query": "query1",
                       "ground_truth": "c0"},
                      ["c2", "c3", "c0"],
                      {1: {"accuracy": 0.0}, 3: {"accuracy": 1.0}},
                      True, 0.0) == results.results[1]
        assert Result({"query": "query2",
                       "ground_truth": "c0"},
                      ["c2", "c3", "c5"],
                      {1: {"accuracy": 0.0}, 3: {"accuracy": 0.0}},
                      True, 0.0) == results.results[2]


@pytest.fixture
def dataset():
    return ListDataset([
        Environment({"query": "query", "ground_truth": "name0"},
                    set(["ground_truth"]))
    ])


@pytest.fixture
def model():
    return MockModel()


@pytest.fixture
def synthesizer(model):
    return MockSynthesizer(model)


class TestEvaluate(object):
    def test_happy_path(self, dataset, model, synthesizer):
        with tempfile.TemporaryDirectory() as tmpdir:
            input = os.path.join(tmpdir, "input")
            output = os.path.join(tmpdir, "output")
            os.makedirs(input)
            os.makedirs(os.path.join(input, "model"))
            torch.save({"score": 1.0, "model": {"score": 1.0, "name": "tmp"}},
                       os.path.join(input, "model", "0"))
            evaluate(input, output, dataset,
                     model, synthesizer,
                     {
                         "accuracy": use_environment(
                             Accuracy(),
                             in_keys=["actual", ["ground_truth", "expected"]],
                             value_key="actual",
                         ),
                         "bleu": use_environment(
                             Bleu(),
                             in_keys=["actual", ["ground_truth", "expected"]],
                             value_key="actual",
                         ),
                     })
            assert os.path.exists(os.path.join(output, "result.pt"))
            assert os.path.exists(
                os.path.join(output, "result_metrics.json"))

    def test_multiple_models(self, dataset, model, synthesizer):
        with tempfile.TemporaryDirectory() as tmpdir:
            input = os.path.join(tmpdir, "input")
            output = os.path.join(tmpdir, "output")
            os.makedirs(input)
            os.makedirs(os.path.join(input, "model"))
            torch.save({"score": 0.5, "model": {"score": 0.5, "name": "tmp"}},
                       os.path.join(input, "model", "0"))
            torch.save({"score": 1.0, "model": {"score": 1.0, "name": "tmp"}},
                       os.path.join(input, "model", "1"))
            evaluate(input, output, dataset,
                     model, synthesizer,
                     {
                         "accuracy": use_environment(
                             Accuracy(),
                             in_keys=["actual", ["ground_truth", "expected"]],
                             value_key="actual",
                         ),
                         "bleu": use_environment(
                             Bleu(),
                             in_keys=["actual", ["ground_truth", "expected"]],
                             value_key="actual",
                         ),
                     })
            assert os.path.exists(os.path.join(output, "result.pt"))
            assert os.path.exists(
                os.path.join(output, "result_metrics.json"))

    def _run(self, init_dir, input, output, model, synthesizer, dataset, rank):
        distributed.initialize(init_dir, rank, 2)
        evaluate(
            input, output, dataset,
            model, synthesizer,
            {
                "accuracy": use_environment(
                    Accuracy(),
                    in_keys=["actual", ["ground_truth", "expected"]],
                    value_key="actual",
                ),
                "bleu": use_environment(
                    Bleu(),
                    in_keys=["actual", ["ground_truth", "expected"]],
                    value_key="actual",
                ),
            }
        )

    def test_multiprocess(self, dataset, model, synthesizer):
        with tempfile.TemporaryDirectory() as tmpdir:
            input = os.path.join(tmpdir, "input")
            output = os.path.join(tmpdir, "output")
            os.makedirs(input)
            os.makedirs(os.path.join(input, "model"))
            torch.save({"score": 0.5, "model": {"score": 0.5, "name": "tmp"}},
                       os.path.join(input, "model", "0"))

            with tempfile.TemporaryDirectory() as init_dir:
                with context.Pool(2) as pool:
                    procs = []
                    for i in range(2):
                        p = pool.apply_async(
                            self._run,
                            args=(init_dir, input, output, model, synthesizer,
                                  dataset, i),
                        )
                        procs.append(p)
                    [p.get() for p in procs]
            assert os.path.exists(os.path.join(output, "result.pt"))
            assert os.path.exists(
                os.path.join(output, "result_metrics.json"))
