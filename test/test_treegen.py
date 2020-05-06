import unittest
import sys
import logging
from dummy_dataset import is_subtype, prepare_dataset
from typing import List
import tempfile
import os

import torch
import fairseq.optim as optim

from nl2prog.gin import nl2prog, treegen, optimizer, workspace
from nl2prog.utils import Query, CommonBeamSearchSynthesizer
from nl2prog.language.action \
    import ast_to_action_sequence as to_seq, ActionOptions
from nl2prog.utils.data \
    import Collate, CollateGroundTruth, collate_none, split_none, \
    CollateNlFeature
from nl2prog.utils.data.treegen \
    import CollateQuery, CollateActionSequence, CollateInput
from nl2prog.utils.transform \
    import TransformDataset, TransformGroundTruth, TransformCode
from nl2prog.utils.transform.treegen import TransformQuery, TransformEvaluator
from nl2prog.nn import Loss
from nl2prog.nn.treegen import TrainModel
from nl2prog.metrics import Accuracy

logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)


def tokenize_query(str: str) -> Query:
    return Query(str.split(" "), str.split(" "))


def tokenize_token(token: str) -> List[str]:
    return [token]


def tokenize_token_2(token: str) -> List[str]:
    if token == "print":
        return [token]
    return list(token)


class TestTreeGen(unittest.TestCase):
    def prepare_encoder(self, to_action_sequence):
        dataset = workspace.get("dataset")["train"]
        treegen.prepare_encoder(dataset, 20, 0, 20, lambda x: x,
                                to_action_sequence,
                                tokenize_query, tokenize_token)

    def prepare_model(self, model_key):
        qencoder = workspace.get("query_encoder")
        cencoder = workspace.get("char_encoder")
        aencoder = workspace.get("action_sequence_encoder")
        model = TrainModel(qencoder, cencoder, aencoder, 32, 2, 4, 1, 6, 5, 5,
                           256, 1024, 0.0)
        workspace.put(model_key, model)

    def prepare_optimizer(self, optimizer_key):
        opt = optimizer.create_optimizer(optim.adafactor.Adafactor,
                                         workspace.get("model"))
        workspace.put(optimizer_key, opt)

    def prepare_synthesizer(self, synthesizer_key, options):
        qencoder = workspace.get("query_encoder")
        cencoder = workspace.get("char_encoder")
        aencoder = workspace.get("action_sequence_encoder")
        self.prepare_model("model")
        model = workspace.get("model")

        transform_input = TransformQuery(tokenize_query, qencoder,
                                         cencoder, 32)
        transform_evaluator = TransformEvaluator(aencoder, 2, 4, train=False)
        synthesizer = CommonBeamSearchSynthesizer(
            5, transform_input, transform_evaluator,
            CollateInput(torch.device("cpu")),
            CollateActionSequence(torch.device("cpu")),
            CollateQuery(torch.device("cpu")), collate_none,
            CollateNlFeature(torch.device("cpu")),
            collate_none, split_none,
            model.input_reader, model.action_sequence_reader, model.decoder,
            model.predictor, aencoder, is_subtype,
            options=options, max_steps=20)
        workspace.put(synthesizer_key, synthesizer)

    def transform_cls(self, to_action_sequence):
        qencoder = workspace.get("query_encoder")
        cencoder = workspace.get("char_encoder")
        aencoder = workspace.get("action_sequence_encoder")
        tquery = TransformQuery(tokenize_query, qencoder, cencoder, 32)
        tcode = TransformCode(to_action_sequence)
        teval = TransformEvaluator(aencoder, 2, 4)
        tgt = TransformGroundTruth(aencoder)
        return TransformDataset(tquery, tcode, teval, tgt)

    def evaluate(self, options, dir):
        with tempfile.TemporaryDirectory() as tmpdir:
            nl2prog.evaluate(
                "dataset", "synthesizer",
                set(["query_encoder", "char_encoder",
                     "action_sequence_encoder"]),
                dir, tmpdir, dir,
                lambda x: prepare_dataset(x, 1),
                lambda key: self.prepare_synthesizer(key, options),
                {"accuracy": Accuracy(lambda x: x, lambda x: x)},
                (5, "accuracy"), top_n=[5]
            )
        results = torch.load(os.path.join(dir, "results.pt"))
        return results["valid"]

    def train(self, options, tokenize_token, output_dir):
        with tempfile.TemporaryDirectory() as tmpdir:
            def to_action_sequence(ast):
                return to_seq(ast, options=options, tokenizer=tokenize_token)

            collate_fn = Collate(
                CollateInput(torch.device("cpu")),
                CollateActionSequence(torch.device("cpu")),
                CollateQuery(torch.device("cpu")),
                CollateGroundTruth(torch.device("cpu")))

            nl2prog.train(
                "dataset", "model", "optimizer",
                set(["query_encoder", "char_encoder",
                     "action_sequence_encoder"]),
                tmpdir, output_dir,
                lambda x: prepare_dataset(x, 10),
                lambda: self.prepare_encoder(to_action_sequence),
                self.prepare_model, self.prepare_optimizer,
                lambda: self.transform_cls(to_action_sequence),
                Loss(), lambda *args: -Loss()(*args), collate_fn, 1, 10,
                num_models=1
            )

    def test_default_settings(self):
        torch.manual_seed(0)
        options = ActionOptions(False, False)
        with tempfile.TemporaryDirectory() as tmpdir:
            self.train(options, tokenize_token, tmpdir)
            results = self.evaluate(options, tmpdir)
        self.assertAlmostEqual(1.0, results.metrics[5]["accuracy"])

    def test_split_nonterminals(self):
        torch.manual_seed(0)
        options = ActionOptions(False, True)
        with tempfile.TemporaryDirectory() as tmpdir:
            self.train(options, tokenize_token, tmpdir)
            results = self.evaluate(options, tmpdir)
        self.assertAlmostEqual(1.0, results.metrics[5]["accuracy"])

        with tempfile.TemporaryDirectory() as tmpdir:
            self.train(options, tokenize_token_2, tmpdir)
            results = self.evaluate(options, tmpdir)
        self.assertAlmostEqual(1.0, results.metrics[5]["accuracy"])

    def test_retain_variadic_args(self):
        torch.manual_seed(0)
        options = ActionOptions(True, False)
        with tempfile.TemporaryDirectory() as tmpdir:
            self.train(options, tokenize_token, tmpdir)
            results = self.evaluate(options, tmpdir)
        self.assertAlmostEqual(1.0, results.metrics[5]["accuracy"])


if __name__ == "__main__":
    unittest.main()
