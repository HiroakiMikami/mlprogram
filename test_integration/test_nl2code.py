from collections import OrderedDict
import logging
import sys
import tempfile
import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchnlp.encoders import LabelEncoder
import mlprogram.nn
from mlprogram.builtins import Pick
from mlprogram.entrypoint import evaluate as eval, train_supervised
from mlprogram.entrypoint import EvaluateSynthesizer
from mlprogram.entrypoint.train import Epoch
from mlprogram.entrypoint.modules.torch import Optimizer
from mlprogram.synthesizers import BeamSearch
from mlprogram.samplers import ActionSequenceSampler
from mlprogram.encoders import ActionSequenceEncoder
from mlprogram.functools import Sequence, Map
from mlprogram.functools import Compose
from mlprogram.utils.data import Collate, CollateOptions
from mlprogram.utils.data import get_words, get_samples
from mlprogram.utils.transform.action_sequence \
    import GroundTruthToActionSequence
from mlprogram.utils.transform.action_sequence import EncodeActionSequence
from mlprogram.utils.transform.action_sequence import AddActions
from mlprogram.utils.transform.action_sequence import AddPreviousActions
from mlprogram.utils.transform.action_sequence import AddHistoryState
from mlprogram.utils.transform.action_sequence import AddStateForRnnDecoder
from mlprogram.utils.transform.text import ExtractReference
from mlprogram.utils.transform.text import EncodeWordQuery
from mlprogram.nn.action_sequence import Loss
import mlprogram.nn.nl2code as nl2code
from mlprogram.metrics import Accuracy

from test_integration.nl2code_dummy_dataset import is_subtype
from test_integration.nl2code_dummy_dataset import train_dataset
from test_integration.nl2code_dummy_dataset import test_dataset
from test_integration.nl2code_dummy_dataset import tokenize
from test_integration.nl2code_dummy_dataset import Parser

logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)


class TestNL2Code(object):
    def prepare_encoder(self, dataset, parser):
        words = get_words(dataset, tokenize)
        samples = get_samples(dataset, parser)
        qencoder = LabelEncoder(words, 2)
        aencoder = ActionSequenceEncoder(samples, 2)
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
        return Optimizer(optim.Adam, model)

    def prepare_synthesizer(self, model, qencoder, aencoder):
        transform_input = Compose(OrderedDict([
            ("extract_reference", ExtractReference(tokenize)),
            ("encode_query", EncodeWordQuery(qencoder))
        ]))
        transform_action_sequence = Compose(OrderedDict([
            ("add_previous_action", AddPreviousActions(aencoder, train=False)),
            ("add_action", AddActions(aencoder, train=False)),
            ("add_state", AddStateForRnnDecoder(train=False)),
            ("add_history", AddHistoryState(train=False))
        ]))
        collate = Collate(
            torch.device("cpu"),
            word_nl_query=CollateOptions(True, 0, -1),
            nl_query_features=CollateOptions(True, 0, -1),
            reference_features=CollateOptions(True, 0, -1),
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
                aencoder, is_subtype, transform_input,
                transform_action_sequence, collate, model))

    def transform_cls(self, qencoder, aencoder, parser):
        tcode = GroundTruthToActionSequence(parser)
        tgt = EncodeActionSequence(aencoder)
        return Sequence(
            OrderedDict([
                ("extract_reference", ExtractReference(tokenize)),
                ("encode_word_query", EncodeWordQuery(qencoder)),
                ("f2", tcode),
                ("add_previous_action", AddPreviousActions(aencoder)),
                ("add_action", AddActions(aencoder)),
                ("add_state", AddStateForRnnDecoder()),
                ("add_history", AddHistoryState()),
                ("f4", tgt)
            ])
        )

    def evaluate(self, qencoder, aencoder, dir):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = self.prepare_model(qencoder, aencoder)
            eval(
                dir, tmpdir, dir,
                test_dataset, model,
                self.prepare_synthesizer(model, qencoder, aencoder),
                {"accuracy": Accuracy()},
                top_n=[5],
                n_process=1
            )
        return torch.load(os.path.join(dir, "result.pt"))

    def train(self, output_dir):
        with tempfile.TemporaryDirectory() as tmpdir:
            loss_fn = nn.Sequential(OrderedDict([
                ("loss", Loss()),
                ("pick",
                 mlprogram.nn.Function(
                     Pick("output@action_sequence_loss")))
            ]))
            collate = Collate(
                torch.device("cpu"),
                word_nl_query=CollateOptions(True, 0, -1),
                nl_query_features=CollateOptions(True, 0, -1),
                reference_features=CollateOptions(True, 0, -1),
                actions=CollateOptions(True, 0, -1),
                previous_actions=CollateOptions(True, 0, -1),
                previous_action_rules=CollateOptions(True, 0, -1),
                history=CollateOptions(False, 1, 0),
                hidden_state=CollateOptions(False, 0, 0),
                state=CollateOptions(False, 0, 0),
                ground_truth_actions=CollateOptions(True, 0, -1)
            ).collate

            qencoder, aencoder = \
                self.prepare_encoder(train_dataset, Parser())
            transform = Map(self.transform_cls(qencoder, aencoder,
                                               Parser()))
            model = self.prepare_model(qencoder, aencoder)
            optimizer = self.prepare_optimizer(model)
            train_supervised(
                tmpdir, output_dir,
                train_dataset, model, optimizer,
                loss_fn,
                EvaluateSynthesizer(
                    test_dataset,
                    self.prepare_synthesizer(model, qencoder, aencoder),
                    {"accuracy": Accuracy()}, top_n=[5]
                ),
                "accuracy@5",
                lambda x: collate(transform(x)),
                1, Epoch(100), evaluation_interval=Epoch(100),
                snapshot_interval=Epoch(100),
                threshold=1.0
            )
        return qencoder, aencoder

    def test(self):
        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as tmpdir:
            encoder = self.train(tmpdir)
            results = self.evaluate(*encoder, tmpdir)
        assert np.allclose(1.0, results.metrics[5]["accuracy"])
