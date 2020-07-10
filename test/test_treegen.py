import unittest
from collections import OrderedDict
import sys
import logging
from dummy_dataset import is_subtype, prepare_dataset
from typing import List
import tempfile
import os

import torch
import fairseq.optim as optim
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
from mlprogram.utils.data import get_words, get_characters, get_samples
from mlprogram.utils.transform import AstToSingleActionSequence
from mlprogram.utils.transform \
    import RandomChoice, TransformGroundTruth, TransformCode
from mlprogram.utils.transform.treegen \
    import TransformQuery, TransformActionSequence
from mlprogram.nn import NL2ProgLoss
from mlprogram.nn import treegen
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


class TestTreeGen(unittest.TestCase):
    def prepare_encoder(self, dataset, to_action_sequence):
        words = get_words(dataset, tokenize_query)
        chars = get_characters(dataset, tokenize_query)
        samples = get_samples(dataset, tokenize_token,
                              to_action_sequence)

        qencoder = LabelEncoder(words, 20)
        cencoder = LabelEncoder(chars, 0)
        aencoder = ActionSequenceEncoder(samples, 20)
        return qencoder, cencoder, aencoder

    def prepare_model(self, qencoder, cencoder, aencoder):

        rule_num = aencoder._rule_encoder.vocab_size
        token_num = aencoder._token_encoder.vocab_size
        node_type_num = aencoder._node_type_encoder.vocab_size
        token_num = aencoder._token_encoder. vocab_size
        return torch.nn.Sequential(OrderedDict([
            ("encoder", treegen.NLReader(
                qencoder.vocab_size, cencoder.vocab_size,
                10, 256, 256, 1, 0.0, 5
            )),
            ("decoder", torch.nn.Sequential(OrderedDict([
                ("action_sequence_reader", treegen.ActionSequenceReader(
                    rule_num, token_num, node_type_num, 4, 256,
                    256, 3, 1, 0.0, 5
                )),
                ("decoder", treegen.Decoder(
                    rule_num, 4, 256, 1024, 256, 1, 0.0, 5
                )),
                ("predictor", treegen.Predictor(
                    256, 256, rule_num, token_num, 256
                ))
            ])))
        ]))

    def prepare_optimizer(self, model):
        return create_optimizer(optim.adafactor.Adafactor,
                                model)

    def prepare_synthesizer(self, qencoder, cencoder, aencoder, options):
        model = self.prepare_model(qencoder, cencoder, aencoder)

        transform_input = TransformQuery(tokenize_query, qencoder,
                                         cencoder, 10)
        transform_action_sequence = TransformActionSequence(aencoder, 4, 4,
                                                            train=False)

        collate = Collate(
            torch.device("cpu"),
            word_nl_query=CollateOptions(True, 0, -1),
            char_nl_query=CollateOptions(True, 0, -1),
            nl_query_features=CollateOptions(True, 0, -1),
            reference_features=CollateOptions(True, 0, -1),
            previous_actions=CollateOptions(True, 0, -1),
            previous_action_rules=CollateOptions(True, 0, -1),
            depthes=CollateOptions(False, 1, 0),
            adjacency_matrix=CollateOptions(False, 0, 0),
            action_queries=CollateOptions(True, 0, -1),
            ground_truth_actions=CollateOptions(True, 0, -1)
        )
        return BeamSearch(
            5, 20,
            ActionSequenceSampler(
                aencoder, lambda x: None, is_subtype, transform_input,
                transform_action_sequence, collate, model, options=options))

    def transform_cls(self, qencoder, cencoder, aencoder, to_action_sequence):
        tquery = TransformQuery(tokenize_query, qencoder, cencoder, 10)
        tcode = TransformCode(to_action_sequence)
        teval = TransformActionSequence(aencoder, 4, 4)
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

    def evaluate(self, qencoder, cencoder, aencoder, options, dir):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = prepare_dataset(1)
            nl2prog.evaluate(
                dir, tmpdir, dir,
                dataset["test"], dataset["valid"],
                self.prepare_synthesizer(
                    qencoder, cencoder, aencoder, options),
                {"accuracy": Accuracy(lambda x: x, lambda x: x)},
                (5, "accuracy"), top_n=[5]
            )
        results = torch.load(os.path.join(dir, "results.pt"))
        return results["valid"]

    def train(self, options, tokenize_token, output_dir):
        with tempfile.TemporaryDirectory() as tmpdir:
            to_action_sequence = \
                AstToSingleActionSequence(options, tokenize_token)

            collate = Collate(
                torch.device("cpu"),
                word_nl_query=CollateOptions(True, 0, -1),
                char_nl_query=CollateOptions(True, 0, -1),
                nl_query_features=CollateOptions(True, 0, -1),
                reference_features=CollateOptions(True, 0, -1),
                previous_actions=CollateOptions(True, 0, -1),
                previous_action_rules=CollateOptions(True, 0, -1),
                depthes=CollateOptions(False, 1, 0),
                adjacency_matrix=CollateOptions(False, 0, 0),
                action_queries=CollateOptions(True, 0, -1),
                ground_truth_actions=CollateOptions(True, 0, -1)
            ).collate

            dataset = prepare_dataset(10)["train"]
            encoder = self.prepare_encoder(dataset, to_action_sequence)
            model = self.prepare_model(*encoder)
            lambda: self.transform_cls(to_action_sequence)
            nl2prog.train(
                tmpdir, output_dir,
                DatasetWithTransform(
                    dataset,
                    self.transform_cls(*encoder, to_action_sequence)),
                model, self.prepare_optimizer(model),
                NL2ProgLoss(), lambda args: -NL2ProgLoss()(args),
                collate,
                1, 10,
                num_models=1
            )
        return encoder

    def test_default_settings(self):
        torch.manual_seed(0)
        options = ActionOptions(False, False)
        with tempfile.TemporaryDirectory() as tmpdir:
            encoder = self.train(options, tokenize_token, tmpdir)
            results = self.evaluate(*encoder, options, tmpdir)
        self.assertAlmostEqual(1.0, results.metrics[5]["accuracy"])

    def test_split_nonterminals(self):
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

    def test_retain_variadic_args(self):
        torch.manual_seed(0)
        options = ActionOptions(True, False)
        with tempfile.TemporaryDirectory() as tmpdir:
            encoder = self.train(options, tokenize_token, tmpdir)
            results = self.evaluate(*encoder, options, tmpdir)
        self.assertAlmostEqual(1.0, results.metrics[5]["accuracy"])


if __name__ == "__main__":
    unittest.main()
