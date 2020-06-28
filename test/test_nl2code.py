import unittest
from collections import OrderedDict
from dummy_dataset import is_subtype, prepare_dataset
from typing import List
import logging
import sys
import tempfile
import os

import torch
import torch.optim as optim
from torchnlp.encoders import LabelEncoder
from mlprogram.entrypoint import nl2prog
from mlprogram.entrypoint.torch import create_optimizer
from mlprogram.utils import Query, Token
from mlprogram.synthesizers import BeamSearch
from mlprogram.samplers import ActionSequenceSampler
from mlprogram.actions import ActionOptions
from mlprogram.encoders import ActionSequenceEncoder
from mlprogram.utils import Sequence
from mlprogram.utils.data import Collate, CollateOptions, DatasetWithTransform
from mlprogram.utils.data import get_words, get_samples
from mlprogram.utils.transform import AstToSingleActionSequence
from mlprogram.utils.transform \
    import RandomChoice, TransformCode, TransformGroundTruth
from mlprogram.utils.transform.nl2code \
    import TransformQuery, TransformActionSequence
from mlprogram.nn import NL2ProgLoss
import mlprogram.nn.nl2code as nl2code
from mlprogram.metrics import Accuracy

logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)


def tokenize_query(str: str) -> Query:
    return Query(
        list(map(lambda x: Token(None, x), str.split(" "))),
        str.split(" "))


def tokenize_token(token: str) -> List[str]:
    return [token]


def tokenize_token_2(token: str) -> List[str]:
    if token == "print":
        return [token]
    return list(token)


class TestNL2Code(unittest.TestCase):
    def prepare_encoder(self, dataset, to_action_sequence):
        words = get_words(dataset, tokenize_query)
        samples = get_samples(dataset, tokenize_token,
                              to_action_sequence)
        qencoder = LabelEncoder(words, 20)
        aencoder = ActionSequenceEncoder(samples, 20)
        return qencoder, aencoder

    def prepare_model(self, qencoder, aencoder):
        reader = nl2code.ActionSequenceReader(
            aencoder._rule_encoder.vocab_size,
            aencoder._token_encoder.vocab_size,
            aencoder._node_type_encoder.vocab_size,
            64, 256
        )
        return torch.nn.Sequential(OrderedDict([
            ("encoder", nl2code.NLReader(qencoder.vocab_size, 256, 256, 0.0)),
            ("decoder",
             torch.nn.Sequential(OrderedDict([
                 ("action_sequence_reader", reader),
                 ("decoder", nl2code.Decoder(256, 2 * 256 + 64, 256, 64, 0.0)),
                 ("predictor", nl2code.Predictor(reader, 256, 256, 256, 64))
             ])))
        ]))

    def prepare_optimizer(self, model):
        return create_optimizer(optim.Adam, model)

    def prepare_synthesizer(self, qencoder, aencoder, model, options):
        transform_input = TransformQuery(tokenize_query, qencoder)
        transform_action_sequence = TransformActionSequence(aencoder,
                                                            train=False)
        collate = Collate(
            torch.device("cpu"),
            word_nl_query=CollateOptions(True, 0, -1),
            nl_query_features=CollateOptions(True, 0, -1),
            actions=CollateOptions(True, 0, -1),
            previous_actions=CollateOptions(True, 0, -1),
            previous_action_rules=CollateOptions(True, 0, -1),
            history=CollateOptions(False, 1, 0),
            hidden_state=CollateOptions(False, 0, 0),
            state=CollateOptions(False, 0, 0),
            ground_truth_actions=CollateOptions(True, 0, -1)
        )
        return BeamSearch(
            5, 20,
            ActionSequenceSampler(
                aencoder, lambda x: None, is_subtype, transform_input,
                transform_action_sequence, collate, model, options=options))

    def transform_cls(self, qencoder, aencoder, to_action_sequence):
        tquery = TransformQuery(tokenize_query, qencoder)
        tcode = TransformCode(to_action_sequence)
        teval = TransformActionSequence(aencoder)
        tgt = TransformGroundTruth(aencoder)
        return Sequence(
            OrderedDict([
                ("f0", RandomChoice()),
                ("f1", tquery),
                ("f2", tcode),
                ("f3", teval),
                ("f4", tgt)
            ])
        )

    def evaluate(self, qencoder, aencoder, options, dir):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = prepare_dataset(1)
            model = self.prepare_model(qencoder, aencoder)
            nl2prog.evaluate(
                dir, tmpdir, dir,
                dataset["test"], dataset["valid"],
                self.prepare_synthesizer(qencoder, aencoder, model, options),
                {"accuracy": Accuracy(lambda x: x, lambda x: x)},
                (5, "accuracy"), top_n=[5]
            )
        results = torch.load(os.path.join(dir, "results.pt"))
        return results["valid"]

    def train(self, options, tokenize_token, output_dir):
        with tempfile.TemporaryDirectory() as tmpdir:
            to_action_sequence = AstToSingleActionSequence(
                options, tokenize_token)
            collate = Collate(
                torch.device("cpu"),
                word_nl_query=CollateOptions(True, 0, -1),
                nl_query_features=CollateOptions(True, 0, -1),
                actions=CollateOptions(True, 0, -1),
                previous_actions=CollateOptions(True, 0, -1),
                previous_action_rules=CollateOptions(True, 0, -1),
                history=CollateOptions(False, 1, 0),
                hidden_state=CollateOptions(False, 0, 0),
                state=CollateOptions(False, 0, 0),
                ground_truth_actions=CollateOptions(True, 0, -1)
            ).collate

            raw_dataset = prepare_dataset(10)["train"]
            qencoder, aencoder = \
                self.prepare_encoder(raw_dataset, to_action_sequence)
            dataset = DatasetWithTransform(
                raw_dataset,
                self.transform_cls(qencoder, aencoder, to_action_sequence))
            model = self.prepare_model(qencoder, aencoder)
            optimizer = self.prepare_optimizer(model)
            nl2prog.train(
                tmpdir, output_dir,
                dataset, model, optimizer,
                NL2ProgLoss(), lambda args: -NL2ProgLoss()(args),
                collate,
                1, 10,
                num_models=1
            )
        return qencoder, aencoder

    def test_default_settings(self):
        torch.manual_seed(0)
        options = ActionOptions(True, True)
        with tempfile.TemporaryDirectory() as tmpdir:
            encoder = self.train(options, tokenize_token, tmpdir)
            results = self.evaluate(*encoder, options, tmpdir)
        self.assertAlmostEqual(1.0, results.metrics[5]["accuracy"])

        with tempfile.TemporaryDirectory() as tmpdir:
            encoder = self.train(options, tokenize_token_2, tmpdir)
            results = self.evaluate(*encoder, options, tmpdir)
        self.assertAlmostEqual(1.0, results.metrics[5]["accuracy"])

    def test_not_split_nonterminals(self):
        torch.manual_seed(0)
        options = ActionOptions(True, False)
        with tempfile.TemporaryDirectory() as tmpdir:
            encoder = self.train(options, tokenize_token, tmpdir)
            results = self.evaluate(*encoder, options, tmpdir)
        self.assertAlmostEqual(1.0, results.metrics[5]["accuracy"])

    def test_not_retain_variadic_args(self):
        torch.manual_seed(0)
        options = ActionOptions(False, True)
        with tempfile.TemporaryDirectory() as tmpdir:
            encoder = self.train(options, tokenize_token, tmpdir)
            results = self.evaluate(*encoder, options, tmpdir)
        self.assertAlmostEqual(1.0, results.metrics[5]["accuracy"])

        with tempfile.TemporaryDirectory() as tmpdir:
            encoder = self.train(options, tokenize_token_2, tmpdir)
            results = self.evaluate(*encoder, options, tmpdir)
        self.assertAlmostEqual(1.0, results.metrics[5]["accuracy"])


if __name__ == "__main__":
    unittest.main()
