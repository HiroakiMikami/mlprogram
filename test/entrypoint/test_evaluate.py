import unittest
import tempfile
import os
import torch

from mlprogram.entrypoint import evaluate
from mlprogram.utils.data import ListDataset
from mlprogram.metrics import Accuracy, Bleu
from mlprogram.entrypoint.evaluate import EvaluateSynthesizer, Result
from mlprogram.synthesizers import Result as DecoderResult


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
    input = input["input"]
    output = []
    if input == "query0":
        output = ["c0", "c1", "c2"]
    elif input == "query1":
        output = ["c2", "c3", "c0"]
    else:
        output = ["c2", "c3", "c5"]

    for i, s in enumerate(output):
        yield DecoderResult(s, -i, True, 1)


class TestEvaluateSynthesizer(unittest.TestCase):
    def test_simple_case(self):
        accuracy = Accuracy()
        dataset = ListDataset([
            {
                "input": "query0",
                "ground_truth": "c0"
            },
            {
                "input": "query1",
                "ground_truth": "c0"
            },
            {
                "input": "query2",
                "ground_truth": "c0"
            }
        ])
        results = EvaluateSynthesizer(dataset, synthesize,
                                      metrics={"accuracy": accuracy})()

        self.assertEqual(
            results.metrics,
            {1: {"accuracy": 1.0 / 3.0}, 3: {"accuracy": 2.0 / 3.0}})
        self.assertEqual(3, len(results.results))
        results.results[0].time = 0.0
        results.results[1].time = 0.0
        results.results[2].time = 0.0
        self.assertEqual(
            Result("query0", {"ground_truth": "c0"},
                   ["c0", "c1", "c2"],
                   {1: {"accuracy": 1.0}, 3: {"accuracy": 1.0}},
                   True, 0.0),
            results.results[0])
        self.assertEqual(
            Result("query1", {"ground_truth": "c0"},
                   ["c2", "c3", "c0"],
                   {1: {"accuracy": 0.0}, 3: {"accuracy": 1.0}},
                   True, 0.0),
            results.results[1])
        self.assertEqual(
            Result("query2", {"ground_truth": "c0"},
                   ["c2", "c3", "c5"],
                   {1: {"accuracy": 0.0}, 3: {"accuracy": 0.0}},
                   True, 0.0),
            results.results[2])

    def test_multiprocess(self):
        accuracy = Accuracy()
        dataset = ListDataset([
            {
                "input": "query0",
                "ground_truth": "c0"
            },
            {
                "input": "query1",
                "ground_truth": "c0"
            },
            {
                "input": "query2",
                "ground_truth": "c0"
            }
        ])
        results = EvaluateSynthesizer(dataset, synthesize,
                                      metrics={"accuracy": accuracy},
                                      n_process=2)()

        self.assertEqual(
            results.metrics,
            {1: {"accuracy": 1.0 / 3}, 3: {"accuracy": 2.0 / 3}})
        self.assertEqual(3, len(results.results))
        results.results[0].time = 0.0
        results.results[1].time = 0.0
        results.results[2].time = 0.0
        results.results.sort(key=lambda x: x.input)
        self.assertEqual(
            Result("query0", {"ground_truth": "c0"},
                   ["c0", "c1", "c2"],
                   {1: {"accuracy": 1.0}, 3: {"accuracy": 1.0}},
                   True, 0.0),
            results.results[0])
        self.assertEqual(
            Result("query1", {"ground_truth": "c0"},
                   ["c2", "c3", "c0"],
                   {1: {"accuracy": 0.0}, 3: {"accuracy": 1.0}},
                   True, 0.0),
            results.results[1])
        self.assertEqual(
            Result("query2", {"ground_truth": "c0"},
                   ["c2", "c3", "c5"],
                   {1: {"accuracy": 0.0}, 3: {"accuracy": 0.0}},
                   True, 0.0),
            results.results[2])


class TestEvaluate(unittest.TestCase):
    def prepare_dataset(self):
        return {"valid": ListDataset([{"input": "query",
                                       "ground_truth": "name0"}])}

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
            evaluate(input, ws, output, dataset["valid"],
                     model, self.prepare_synthesizer(model),
                     {
                "accuracy": Accuracy(),
                "bleu": Bleu(),
            })
            self.assertTrue(os.path.exists(os.path.join(output, "result.pt")))
            self.assertTrue(os.path.exists(
                os.path.join(output, "result_metrics.json")))

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
            evaluate(input, ws, output, dataset["valid"],
                     model, self.prepare_synthesizer(model),
                     {
                "accuracy": Accuracy(),
                "bleu": Bleu(),
            })
            self.assertTrue(os.path.exists(os.path.join(output, "result.pt")))
            self.assertTrue(os.path.exists(
                os.path.join(output, "result_metrics.json")))

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
            evaluate(input, ws, output, dataset["valid"],
                     model, self.prepare_synthesizer(model),
                     {
                "accuracy": Accuracy(),
                "bleu": Bleu(),
            }, n_process=2)
            self.assertTrue(os.path.exists(os.path.join(output, "result.pt")))
            self.assertTrue(os.path.exists(
                os.path.join(output, "result_metrics.json")))


if __name__ == "__main__":
    unittest.main()
