import os
import tempfile

import torch

from mlprogram.builtins import Environment
from mlprogram.entrypoint import evaluate
from mlprogram.entrypoint.evaluate import EvaluateSynthesizer, Result
from mlprogram.metrics import Accuracy, Bleu
from mlprogram.synthesizers import Result as DecoderResult
from mlprogram.utils.data import ListDataset


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
        accuracy = Accuracy()
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

    def test_multiprocess(self):
        accuracy = Accuracy()
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
                                      metrics={"accuracy": accuracy},
                                      n_process=2)()

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


class TestEvaluate(object):
    def prepare_dataset(self):
        return ListDataset([
            Environment({"query": "query", "ground_truth": "name0"},
                        set(["ground_truth"]))
        ])

    def prepare_model(self):
        return MockModel()

    def prepare_synthesizer(self, model):
        return MockSynthesizer(model)

    def test_happy_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input = os.path.join(tmpdir, "input")
            ws = os.path.join(tmpdir, "workspace")
            output = os.path.join(tmpdir, "output")
            os.makedirs(input)
            os.makedirs(os.path.join(input, "model"))
            torch.save({"score": 1.0, "model": {"score": 1.0, "name": "tmp"}},
                       os.path.join(input, "model", "0"))
            dataset = self.prepare_dataset()
            model = self.prepare_model()
            evaluate(input, ws, output, dataset,
                     model, self.prepare_synthesizer(model),
                     {
                         "accuracy": Accuracy(),
                         "bleu": Bleu(),
                     })
            assert os.path.exists(os.path.join(output, "result.pt"))
            assert os.path.exists(
                os.path.join(output, "result_metrics.json"))

    def test_multiple_models(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input = os.path.join(tmpdir, "input")
            ws = os.path.join(tmpdir, "workspace")
            output = os.path.join(tmpdir, "output")
            os.makedirs(input)
            os.makedirs(os.path.join(input, "model"))
            torch.save({"score": 0.5, "model": {"score": 0.5, "name": "tmp"}},
                       os.path.join(input, "model", "0"))
            torch.save({"score": 1.0, "model": {"score": 1.0, "name": "tmp"}},
                       os.path.join(input, "model", "1"))
            dataset = self.prepare_dataset()
            model = self.prepare_model()
            evaluate(input, ws, output, dataset,
                     model, self.prepare_synthesizer(model),
                     {
                         "accuracy": Accuracy(),
                         "bleu": Bleu(),
                     })
            assert os.path.exists(os.path.join(output, "result.pt"))
            assert os.path.exists(
                os.path.join(output, "result_metrics.json"))

    def test_multiprocess(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input = os.path.join(tmpdir, "input")
            ws = os.path.join(tmpdir, "workspace")
            output = os.path.join(tmpdir, "output")
            os.makedirs(input)
            os.makedirs(os.path.join(input, "model"))
            torch.save({"score": 0.5, "model": {"score": 0.5, "name": "tmp"}},
                       os.path.join(input, "model", "0"))
            dataset = self.prepare_dataset()
            model = self.prepare_model()
            evaluate(input, ws, output, dataset,
                     model, self.prepare_synthesizer(model),
                     {
                         "accuracy": Accuracy(),
                         "bleu": Bleu(),
                     }, n_process=2)
            assert os.path.exists(os.path.join(output, "result.pt"))
            assert os.path.exists(
                os.path.join(output, "result_metrics.json"))
