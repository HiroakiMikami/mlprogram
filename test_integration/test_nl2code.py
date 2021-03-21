import logging
import os
import sys
import tempfile
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchnlp.encoders import LabelEncoder

import mlprogram.nn
import mlprogram.nn.nl2code as nl2code
from mlprogram.builtins import Apply, Pick
from mlprogram.encoders import ActionSequenceEncoder
from mlprogram.entrypoint import EvaluateSynthesizer
from mlprogram.entrypoint import evaluate as eval
from mlprogram.entrypoint import train_supervised
from mlprogram.entrypoint.modules.torch import Optimizer
from mlprogram.entrypoint.train import Epoch
from mlprogram.functools import Compose, Map, Sequence
from mlprogram.metrics import Accuracy, use_environment
from mlprogram.nn.action_sequence import Loss
from mlprogram.samplers import ActionSequenceSampler
from mlprogram.synthesizers import BeamSearch
from mlprogram.transforms.action_sequence import (
    AddActions,
    AddPreviousActions,
    AddState,
    EncodeActionSequence,
    GroundTruthToActionSequence,
)
from mlprogram.transforms.text import EncodeWordQuery
from mlprogram.utils.data import Collate, CollateOptions, get_samples, get_words
from test_integration.nl2code_dummy_dataset import (
    Parser,
    is_subtype,
    test_dataset,
    tokenize,
    train_dataset,
)

logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)


class TestNL2Code(object):
    def prepare_encoder(self, dataset, parser):
        words = get_words(dataset, tokenize)
        samples = get_samples(dataset, parser)
        qencoder = LabelEncoder(words, 2)
        aencoder = ActionSequenceEncoder(samples, 2)
        return qencoder, aencoder

    def prepare_model(self, qencoder, aencoder):
        embedding = mlprogram.nn.action_sequence.ActionsEmbedding(
            aencoder._rule_encoder.vocab_size,
            aencoder._token_encoder.vocab_size,
            aencoder._node_type_encoder.vocab_size,
            64, 256
        )
        decoder = nl2code.Decoder(
            embedding.output_size, 256, 256, 64, 0.0
        )
        return torch.nn.Sequential(OrderedDict([
            ("encoder",
             torch.nn.Sequential(OrderedDict([
                 ("embedding",
                  Apply(
                      module=mlprogram.nn.EmbeddingWithMask(
                          qencoder.vocab_size, 256, -1
                      ),
                      in_keys=[["word_nl_query", "x"]],
                      out_key="nl_features"
                  )),
                 ("lstm",
                  Apply(
                      module=mlprogram.nn.BidirectionalLSTM(
                          256, 256, 0.0),
                      in_keys=[["nl_features", "x"]],
                      out_key="reference_features"
                  )),
             ]))),
            ("decoder",
             torch.nn.Sequential(OrderedDict([
                 ("embedding",
                  Apply(
                      module=embedding,
                      in_keys=[
                          "actions",
                          "previous_actions",
                      ],
                      out_key="action_features"
                  )),
                 ("decoder",
                  Apply(
                      module=decoder,
                      in_keys=[
                          ["reference_features", "nl_query_features"],
                          "actions",
                          "action_features",
                          "history",
                          "hidden_state",
                          "state",
                      ],
                      out_key=[
                          "action_features",
                          "action_contexts",
                          "history",
                          "hidden_state",
                          "state",
                      ]
                  )),
                 ("predictor",
                  Apply(
                      module=nl2code.Predictor(embedding, 256, 256, 256, 64),
                      in_keys=["reference_features",
                               "action_features", "action_contexts"],
                      out_key=["rule_probs", "token_probs", "reference_probs"],
                  ))
             ])))
        ]))

    def prepare_optimizer(self, model):
        return Optimizer(optim.Adam, model)

    def prepare_synthesizer(self, model, qencoder, aencoder):
        transform_input = Compose(OrderedDict([
            ("extract_reference", Apply(
                module=mlprogram.nn.Function(tokenize),
                in_keys=[["text_query", "str"]], out_key="reference")),
            ("encode_query", Apply(
                module=EncodeWordQuery(qencoder),
                in_keys=["reference"],
                out_key="word_nl_query"))
        ]))
        transform_action_sequence = Compose(OrderedDict([
            ("add_previous_action",
             Apply(
                 module=AddPreviousActions(aencoder, n_dependent=1),
                 in_keys=["action_sequence", "reference"],
                 constants={"train": False},
                 out_key="previous_actions",
             )),
            ("add_action",
             Apply(
                 module=AddActions(aencoder, n_dependent=1),
                 in_keys=["action_sequence", "reference"],
                 constants={"train": False},
                 out_key="actions",
             )),
            ("add_state", AddState("state")),
            ("add_hidden_state", AddState("hidden_state")),
            ("add_history", AddState("history"))
        ]))
        collate = Collate(
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
        return Sequence(
            OrderedDict([
                ("extract_reference", Apply(
                    module=mlprogram.nn.Function(tokenize),
                    in_keys=[["text_query", "str"]], out_key="reference")),
                ("encode_word_query", Apply(
                    module=EncodeWordQuery(qencoder),
                    in_keys=["reference"],
                    out_key="word_nl_query")),
                ("f2",
                 Apply(
                     module=GroundTruthToActionSequence(parser),
                     in_keys=["ground_truth"],
                     out_key="action_sequence",
                 )),
                ("add_previous_action",
                 Apply(
                     module=AddPreviousActions(aencoder, n_dependent=1),
                     in_keys=["action_sequence", "reference"],
                     constants={"train": True},
                     out_key="previous_actions",
                 )),
                ("add_action",
                 Apply(
                     module=AddActions(aencoder, n_dependent=1),
                     in_keys=["action_sequence", "reference"],
                     constants={"train": True},
                     out_key="actions",
                 )),
                ("add_state", AddState("state")),
                ("add_hidden_state", AddState("hidden_state")),
                ("add_history", AddState("history")),
                ("f4",
                 Apply(
                     module=EncodeActionSequence(aencoder),
                     in_keys=["action_sequence", "reference"],
                     out_key="ground_truth_actions",
                 ))
            ])
        )

    def evaluate(self, qencoder, aencoder, dir):
        model = self.prepare_model(qencoder, aencoder)
        eval(
            dir, dir,
            test_dataset, model,
            self.prepare_synthesizer(model, qencoder, aencoder),
            {"accuracy": use_environment(
                metric=Accuracy(),
                in_keys=["actual", ["ground_truth", "expected"]],
                value_key="actual"
            )},
            top_n=[5],
        )
        return torch.load(os.path.join(dir, "result.pt"))

    def train(self, output_dir):
        loss_fn = nn.Sequential(OrderedDict([
            ("loss",
             Apply(
                 module=Loss(),
                 in_keys=[
                     "rule_probs",
                     "token_probs",
                     "reference_probs",
                     "ground_truth_actions",
                 ],
                 out_key="action_sequence_loss",
             )),
            ("pick",
             mlprogram.nn.Function(
                 Pick("action_sequence_loss")))
        ]))
        collate = Collate(
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

        qencoder, aencoder = self.prepare_encoder(train_dataset, Parser())
        transform = Map(self.transform_cls(qencoder, aencoder, Parser()))
        model = self.prepare_model(qencoder, aencoder)
        optimizer = self.prepare_optimizer(model)
        train_supervised(
            output_dir,
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
