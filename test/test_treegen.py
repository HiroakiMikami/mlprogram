import unittest
from collections import OrderedDict
import sys
import logging
from dummy_dataset import is_subtype, train_dataset, test_dataset
import tempfile
import os

import torch
import fairseq.optim as optim
from torchnlp.encoders import LabelEncoder

from mlprogram.entrypoint import evaluate as eval, train_supervised
from mlprogram.entrypoint.train import Epoch
from mlprogram.entrypoint.torch import Optimizer
from mlprogram.utils import Query, Token
from mlprogram.synthesizers import BeamSearch
from mlprogram.samplers import ActionSequenceSampler
from mlprogram.encoders import ActionSequenceEncoder
from mlprogram.utils import Sequence, Map
from mlprogram.utils.data import Collate, CollateOptions
from mlprogram.utils.data import get_words, get_characters, get_samples
from mlprogram.utils.transform import AstToSingleActionSequence
from mlprogram.utils.transform import RandomChoice
from mlprogram.utils.transform.action_sequence \
    import TransformGroundTruth, TransformCode
from mlprogram.utils.transform.treegen \
    import TransformQuery, TransformActionSequence
from mlprogram.nn.action_sequence import Loss, Predictor
from mlprogram.nn import treegen
from mlprogram.metrics import Accuracy

logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)


def tokenize_query(str: str) -> Query:
    return Query(
        list(map(lambda x: Token(None, x), str.split(" "))),
        str.split(" "))


class TestTreeGen(unittest.TestCase):
    def prepare_encoder(self, dataset, to_action_sequence):
        words = get_words(dataset, tokenize_query)
        chars = get_characters(dataset, tokenize_query)
        samples = get_samples(dataset, to_action_sequence)

        qencoder = LabelEncoder(words, 2)
        cencoder = LabelEncoder(chars, 0)
        aencoder = ActionSequenceEncoder(samples, 2)
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
                ("predictor", Predictor(
                    256, 256, rule_num, token_num, 256
                ))
            ])))
        ]))

    def prepare_optimizer(self, model):
        return Optimizer(optim.adafactor.Adafactor,
                         model)

    def prepare_synthesizer(self, model, qencoder, cencoder, aencoder):
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
                transform_action_sequence, collate, model))

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

    def evaluate(self, qencoder, cencoder, aencoder, dir):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = self.prepare_model(qencoder, cencoder, aencoder)
            eval(
                dir, tmpdir, dir,
                test_dataset, test_dataset,
                model,
                self.prepare_synthesizer(
                    model, qencoder, cencoder, aencoder),
                {"accuracy": Accuracy()},
                (5, "accuracy"), top_n=[5]
            )
        results = torch.load(os.path.join(dir, "results.pt"))
        return results["valid"]

    def train(self, output_dir):
        with tempfile.TemporaryDirectory() as tmpdir:
            to_action_sequence = AstToSingleActionSequence()

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

            encoder = self.prepare_encoder(train_dataset, to_action_sequence)
            model = self.prepare_model(*encoder)
            transform = Map(self.transform_cls(*encoder, to_action_sequence))
            train_supervised(
                tmpdir, output_dir, train_dataset,
                model, self.prepare_optimizer(model),
                Loss(), lambda args: -Loss()(args),
                lambda x: collate(transform(x)),
                1, Epoch(100), interval=Epoch(10),
                num_models=1
            )
        return encoder

    def test(self):
        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as tmpdir:
            encoder = self.train(tmpdir)
            results = self.evaluate(*encoder, tmpdir)
        self.assertAlmostEqual(1.0, results.metrics[5]["accuracy"])


if __name__ == "__main__":
    unittest.main()
