import logging
import os
import random
import sys
import tempfile
from collections import OrderedDict

import fairseq.optim as optim
import numpy as np
import torch
import torch.nn as nn
from torchnlp.encoders import LabelEncoder

import mlprogram.nn
from mlprogram.builtins import Apply, Pick
from mlprogram.encoders import ActionSequenceEncoder
from mlprogram.entrypoint import EvaluateSynthesizer
from mlprogram.entrypoint import evaluate as eval
from mlprogram.entrypoint import train_supervised
from mlprogram.entrypoint.modules.torch import Optimizer
from mlprogram.entrypoint.train import Epoch
from mlprogram.functools import Compose, Map, Sequence
from mlprogram.metrics import Accuracy, use_environment
from mlprogram.nn import treegen
from mlprogram.nn.action_sequence import Loss, Predictor
from mlprogram.samplers import ActionSequenceSampler
from mlprogram.synthesizers import BeamSearch
from mlprogram.transforms.action_sequence import (
    AddActionSequenceAsTree,
    AddPreviousActionRules,
    AddPreviousActions,
    AddQueryForTreeGenDecoder,
    EncodeActionSequence,
    GroundTruthToActionSequence,
)
from mlprogram.transforms.text import EncodeCharacterQuery, EncodeWordQuery
from mlprogram.utils.data import (
    Collate,
    CollateOptions,
    get_characters,
    get_samples,
    get_words,
)
from test_integration.nl2code_dummy_dataset import (
    Parser,
    is_subtype,
    test_dataset,
    tokenize,
    train_dataset,
)

logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)


class TestTreeGen(object):
    def prepare_encoder(self, dataset, parser):
        words = get_words(dataset, tokenize)
        chars = get_characters(dataset, tokenize)
        samples = get_samples(dataset, parser)

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
            ("encoder", Apply(
                module=treegen.Encoder(
                    qencoder.vocab_size, cencoder.vocab_size,
                    10, 256, 256, 1, 0.0, 5
                ),
                in_keys=["word_nl_query", "char_nl_query"],
                out_key="reference_features"
            )),
            ("decoder", torch.nn.Sequential(OrderedDict([
                ("decoder",
                 Apply(
                     module=treegen.Decoder(
                         n_rule=rule_num, n_token=token_num, n_node_type=node_type_num,
                         max_depth=4, max_arity=4,
                         rule_embedding_size=256, encoder_hidden_size=256,
                         decoder_hidden_size=1024, out_size=256,
                         tree_conv_kernel_size=3, n_head=1, dropout=0.0,
                         n_encoder_block=5, n_decoder_block=5,
                     ),
                     in_keys=[["reference_features", "nl_query_features"],
                              "action_queries",
                              "previous_actions",
                              "previous_action_rules",
                              "depthes",
                              "adjacency_matrix"],
                     out_key="action_features")),
                ("predictor",
                 Apply(
                     module=Predictor(256, 256, rule_num, token_num, 256),
                     in_keys=["reference_features", "action_features"],
                     out_key=["rule_probs", "token_probs", "reference_probs"]
                 ))
            ])))
        ]))

    def prepare_optimizer(self, model):
        return Optimizer(optim.adafactor.Adafactor,
                         model)

    def prepare_synthesizer(self, model, qencoder, cencoder, aencoder):
        transform_input = Compose(OrderedDict([
            ("extract_reference", Apply(
                module=mlprogram.nn.Function(tokenize),
                in_keys=[["text_query", "str"]], out_key="reference")),
            ("encode_word_query", Apply(
                module=EncodeWordQuery(qencoder),
                in_keys=["reference"],
                out_key="word_nl_query")),
            ("encode_char", Apply(
                module=EncodeCharacterQuery(cencoder, 10),
                in_keys=["reference"],
                out_key="char_nl_query"))
        ]))
        transform_action_sequence = Compose(OrderedDict([
            ("add_previous_action",
             Apply(
                 module=AddPreviousActions(aencoder),
                 in_keys=["action_sequence", "reference"],
                 constants={"train": False},
                 out_key="previous_actions",
             )),
            ("add_previous_action_rule",
             Apply(
                 module=AddPreviousActionRules(aencoder, 4,),
                 in_keys=["action_sequence", "reference"],
                 constants={"train": False},
                 out_key="previous_action_rules",
             )),
            ("add_tree",
             Apply(
                 module=AddActionSequenceAsTree(aencoder),
                 in_keys=["action_sequence", "reference"],
                 constants={"train": False},
                 out_key=["adjacency_matrix", "depthes"],
             )),
            ("add_query",
             Apply(
                 module=AddQueryForTreeGenDecoder(aencoder, 4),
                 in_keys=["action_sequence", "reference"],
                 constants={"train": False},
                 out_key="action_queries",
             ))
        ]))

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
                aencoder, is_subtype, transform_input,
                transform_action_sequence, collate, model))

    def transform_cls(self, qencoder, cencoder, aencoder, parser):
        return Sequence(
            OrderedDict([
                ("extract_reference", Apply(
                    module=mlprogram.nn.Function(tokenize),
                    in_keys=[["text_query", "str"]], out_key="reference")),
                ("encode_word_query", Apply(
                    module=EncodeWordQuery(qencoder),
                    in_keys=["reference"],
                    out_key="word_nl_query")),
                ("encode_char", Apply(
                    module=EncodeCharacterQuery(cencoder, 10),
                    in_keys=["reference"],
                    out_key="char_nl_query")),
                ("f2",
                 Apply(
                     module=GroundTruthToActionSequence(parser),
                     in_keys=["ground_truth"],
                     out_key="action_sequence",
                 )),
                ("add_previous_action",
                 Apply(
                     module=AddPreviousActions(aencoder),
                     in_keys=["action_sequence", "reference"],
                     constants={"train": True},
                     out_key="previous_actions",
                 )),
                ("add_previous_action_rule",
                 Apply(
                     module=AddPreviousActionRules(aencoder, 4),
                     in_keys=["action_sequence", "reference"],
                     constants={"train": True},
                     out_key="previous_action_rules",
                 )),
                ("add_tree",
                 Apply(
                     module=AddActionSequenceAsTree(aencoder),
                     in_keys=["action_sequence", "reference"],
                     constants={"train": True},
                     out_key=["adjacency_matrix", "depthes"],
                 )),
                ("add_query",
                 Apply(
                     module=AddQueryForTreeGenDecoder(aencoder, 4),
                     in_keys=["action_sequence", "reference"],
                     constants={"train": True},
                     out_key="action_queries",
                 )),
                ("f4",
                 Apply(
                     module=EncodeActionSequence(aencoder),
                     in_keys=["action_sequence", "reference"],
                     out_key="ground_truth_actions",
                 ))
            ])
        )

    def evaluate(self, qencoder, cencoder, aencoder, dir):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = self.prepare_model(qencoder, cencoder, aencoder)
            eval(
                dir, tmpdir, dir,
                test_dataset,
                model,
                self.prepare_synthesizer(
                    model, qencoder, cencoder, aencoder),
                {"accuracy": use_environment(
                    metric=Accuracy(),
                    in_keys=["actual", ["ground_truth", "expected"]],
                    value_key="actual"
                )},
                top_n=[5],
            )
        return torch.load(os.path.join(dir, "result.pt"))

    def train(self, output_dir):
        with tempfile.TemporaryDirectory() as tmpdir:
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

            encoder = self.prepare_encoder(train_dataset, Parser())
            model = self.prepare_model(*encoder)
            transform = Map(self.transform_cls(*encoder, Parser()))
            train_supervised(
                tmpdir, output_dir, train_dataset,
                model, self.prepare_optimizer(model),
                loss_fn,
                EvaluateSynthesizer(
                    test_dataset,
                    self.prepare_synthesizer(model, *encoder),
                    {"accuracy": Accuracy()},
                    top_n=[5]
                ),
                "accuracy@5",
                lambda x: collate(transform(x)),
                1, Epoch(100), evaluation_interval=Epoch(100),
                snapshot_interval=Epoch(100),
                threshold=1.0
            )
        return encoder

    def test(self):
        torch.manual_seed(1)
        np.random.seed(1)
        random.seed(1)
        with tempfile.TemporaryDirectory() as tmpdir:
            encoder = self.train(tmpdir)
            results = self.evaluate(*encoder, tmpdir)
        assert np.allclose(1.0, results.metrics[5]["accuracy"])
