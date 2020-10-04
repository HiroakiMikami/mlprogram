import numpy as np
import unittest
from collections import OrderedDict
import logging
import sys
import tempfile
import os
import random

import torch
import torch.optim as optim
from mlprogram.entrypoint \
    import train_supervised, train_REINFORCE, evaluate as eval
from mlprogram.entrypoint import EvaluateSynthesizer
from mlprogram.entrypoint.train import Epoch
from mlprogram.entrypoint.modules.torch import Optimizer, Reshape
from mlprogram.synthesizers \
    import SMC, FilteredSynthesizer, SynthesizerWithTimeout
from mlprogram.samplers import ActionSequenceSampler
from mlprogram.samplers import SequentialProgramSampler
from mlprogram.samplers import SamplerWithValueNetwork
from mlprogram.samplers import FilteredSampler
from mlprogram.encoders import ActionSequenceEncoder
from mlprogram.utils import Sequence, Map, Flatten, Compose, Threshold, Pick
from mlprogram.utils.data import Collate, CollateOptions
import mlprogram.nn
from mlprogram.nn.action_sequence import Loss
from mlprogram.nn import CNN2d, Apply, AggregatedLoss, MLP
from mlprogram.nn.pbe_with_repl import Encoder
import mlprogram.nn.action_sequence as a_s
from mlprogram import metrics
from mlprogram.languages.csg import get_samples, IsSubtype
from mlprogram.languages.csg import Interpreter, Parser, Dataset
from mlprogram.utils.data \
    import to_map_style_dataset, transform as data_transform
from mlprogram.languages.csg.transform import TransformCanvas
from mlprogram.languages.csg.transform import AddTestCases
from mlprogram.utils.transform.action_sequence \
    import TransformCode, TransformGroundTruth, \
    TransformActionSequenceForRnnDecoder
from mlprogram.utils.transform.pbe_with_repl import ToEpisode, EvaluateCode
from test_case_utils import integration_test

logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)


class TestCsgByPbeWithREPL(unittest.TestCase):
    def prepare_encoder(self, dataset, parser):
        return ActionSequenceEncoder(get_samples(dataset, parser),
                                     0)

    def prepare_model(self, encoder: ActionSequenceEncoder):
        return torch.nn.Sequential(OrderedDict([
            ("encode_input",
             Apply([("processed_input", "x")], "input_feature",
                   CNN2d(1, 16, 32, 2, 2, 2))),
            ("encoder",
             Encoder(CNN2d(2, 16, 32, 2, 2, 2))),
            ("decoder",
             torch.nn.Sequential(OrderedDict([
                 ("action_sequence_reader",
                  a_s.ActionSequenceReader(encoder._rule_encoder.vocab_size,
                                           encoder._token_encoder.vocab_size,
                                           256)),
                 ("decoder",
                  a_s.RnnDecoder(2 * 16 * 8 * 8, 256, 512, 0.0)),
                 ("predictor",
                  a_s.Predictor(512, 16 * 8 * 8,
                                encoder._rule_encoder.vocab_size,
                                encoder._token_encoder.vocab_size,
                                512))
             ]))),
            ("value",
             Apply([("variable_feature", "x")], "value",
                   MLP(16 * 8 * 8, 1, 512, 2,
                       activation=torch.nn.Sigmoid()),
                   value_type="tensor"))
        ]))

    def prepare_optimizer(self, model):
        return Optimizer(optim.Adam, model)

    def prepare_synthesizer(self, model, encoder, interpreter, rollout=True):
        collate = Collate(
            torch.device("cpu"),
            processed_input=CollateOptions(False, 0, 0),
            input_feature=CollateOptions(False, 0, 0),
            reference_features=CollateOptions(True, 0, 0),
            variables=CollateOptions(True, 0, 0),
            previous_actions=CollateOptions(True, 0, -1),
            hidden_state=CollateOptions(False, 0, 0),
            state=CollateOptions(False, 0, 0),
            ground_truth_actions=CollateOptions(True, 0, -1)
        )
        subsampler = ActionSequenceSampler(
            encoder, IsSubtype(),
            Compose(OrderedDict([
                ("ecode", EvaluateCode(interpreter)),
                ("tcanvas", TransformCanvas(["variables"]))
            ])),
            TransformActionSequenceForRnnDecoder(encoder, train=False),
            collate, model,
            rng=np.random.RandomState(0))
        subsynthesizer = SMC(
            5, 1,
            subsampler,
            max_try_num=1,
            to_key=Pick("action_sequence"),
            rng=np.random.RandomState(0)
        )

        sampler = SequentialProgramSampler(
            subsynthesizer,
            TransformCanvas(["input"]),
            collate,
            model.encode_input,
            to_code=Parser().unparse,
            rng=np.random.RandomState(0))
        if rollout:
            sampler = FilteredSampler(
                sampler,
                metrics.TestCaseResult(interpreter, reference=True,
                                       metric=metrics.Iou()),
                0.9
            )
            return SMC(4, 20, sampler, rng=np.random.RandomState(0),
                       to_key=Pick("code"), max_try_num=1)
        else:
            sampler = SamplerWithValueNetwork(
                sampler,
                Compose(OrderedDict([
                    ("ecode", EvaluateCode(interpreter)),
                    ("tcanvas", TransformCanvas(["variables"]))
                ])),
                collate,
                torch.nn.Sequential(OrderedDict([
                    ("encoder", model.encoder),
                    ("value", model.value),
                    ("pick",
                     mlprogram.nn.Function(mlprogram.utils.Pick("value")))
                ])))

            synthesizer = SynthesizerWithTimeout(
                SMC(4, 20, sampler, rng=np.random.RandomState(0),
                    to_key=Pick("code")),
                1
            )
            return FilteredSynthesizer(
                synthesizer,
                metrics.TestCaseResult(interpreter, reference=True,
                                       metric=metrics.Iou()),
                0.9
            )

    def interpreter(self):
        return Interpreter(2, 2, 8)

    def to_episode(self, encoder, interpreter):
        return ToEpisode(Parser().parse, remove_used_reference=True)

    def transform(self, encoder, interpreter, parser):
        tcanvas = TransformCanvas(["input", "variables"])
        tcode = TransformCode(parser)
        taction = TransformActionSequenceForRnnDecoder(encoder)
        tgt = TransformGroundTruth(encoder)
        return Sequence(
            OrderedDict([
                ("evaluate_code", EvaluateCode(interpreter)),
                ("tcanvas", tcanvas),
                ("tcode", tcode),
                ("teval", taction),
                ("tgt", tgt)
            ])
        )

    def evaluate(self, dataset, encoder, dir):
        with tempfile.TemporaryDirectory() as tmpdir:
            interpreter = self.interpreter()
            model = self.prepare_model(encoder)
            eval(
                dir, tmpdir, dir,
                dataset,
                model,
                self.prepare_synthesizer(model, encoder, interpreter,
                                         rollout=False),
                {}, top_n=[],
                n_process=1
            )
        return torch.load(os.path.join(dir, "result.pt"))

    def pretrain(self, output_dir):
        dataset = Dataset(2, 1, 2, 1, 45, reference=True, seed=1)
        train_dataset = to_map_style_dataset(dataset, 10)

        with tempfile.TemporaryDirectory() as tmpdir:
            interpreter = self.interpreter()
            train_dataset = data_transform(
                train_dataset,
                AddTestCases(interpreter, reference=True))
            encoder = self.prepare_encoder(dataset, Parser())

            collate = Collate(
                torch.device("cpu"),
                processed_input=CollateOptions(False, 0, 0),
                variables=CollateOptions(True, 0, 0),
                previous_actions=CollateOptions(True, 0, -1),
                hidden_state=CollateOptions(False, 0, 0),
                state=CollateOptions(False, 0, 0),
                ground_truth_actions=CollateOptions(True, 0, -1)
            )
            collate_fn = Sequence(OrderedDict([
                ("to_episode", Map(self.to_episode(encoder,
                                                   interpreter))),
                ("flatten", Flatten()),
                ("transform", Map(self.transform(
                    encoder, interpreter, Parser()))),
                ("collate", collate)
            ]))

            model = self.prepare_model(encoder)
            optimizer = self.prepare_optimizer(model)
            train_supervised(
                tmpdir, output_dir,
                train_dataset, model, optimizer,
                torch.nn.Sequential(OrderedDict([
                    ("loss", Loss(reduction="sum")),
                    ("normalize",  # divided by batch_size
                     Apply(
                         [("action_sequence_loss", "lhs")],
                         "action_sequence_loss",
                         mlprogram.nn.Function(mlprogram.utils.Div()),
                         constants={"rhs": 1})),
                    ("pick",
                     mlprogram.nn.Function(
                         mlprogram.utils.Pick("action_sequence_loss")))
                ])),
                None, "score",
                collate_fn,
                1, Epoch(100), evaluation_interval=Epoch(10),
                snapshot_interval=Epoch(100)
            )
        return encoder, train_dataset

    def reinforce(self, train_dataset, encoder, output_dir):
        with tempfile.TemporaryDirectory() as tmpdir:
            interpreter = self.interpreter()

            collate = Collate(
                torch.device("cpu"),
                processed_input=CollateOptions(False, 0, 0),
                variables=CollateOptions(True, 0, 0),
                previous_actions=CollateOptions(True, 0, -1),
                hidden_state=CollateOptions(False, 0, 0),
                state=CollateOptions(False, 0, 0),
                ground_truth_actions=CollateOptions(True, 0, -1),
                reward=CollateOptions(False, 0, 0)
            )
            collate_fn = Sequence(OrderedDict([
                ("to_episode", Map(self.to_episode(encoder,
                                                   interpreter))),
                ("flatten", Flatten()),
                ("transform", Map(self.transform(
                    encoder, interpreter, Parser()))),
                ("collate", collate)
            ]))

            model = self.prepare_model(encoder)
            optimizer = self.prepare_optimizer(model)
            train_REINFORCE(
                output_dir, tmpdir, output_dir,
                train_dataset,
                self.prepare_synthesizer(model, encoder, interpreter),
                model, optimizer,
                torch.nn.Sequential(OrderedDict([
                    ("policy",
                     torch.nn.Sequential(OrderedDict([
                         ("loss", Loss(reduction="none")),
                         ("weight_by_reward",
                             Apply(
                                 [("reward", "lhs"),
                                  ("action_sequence_loss", "rhs")],
                                 "action_sequence_loss",
                                 mlprogram.nn.Function(mlprogram.utils.Mul())))
                     ]))),
                    ("value",
                     torch.nn.Sequential(OrderedDict([
                         ("reshape_reward",
                             Apply(
                                 [("reward", "x")],
                                 "value_loss_target",
                                 Reshape([-1, 1]))),
                         ("BCE",
                             Apply(
                                 [("value", "input"),
                                  ("value_loss_target", "target")],
                                 "value_loss",
                                 torch.nn.BCELoss(reduction='sum')))
                     ]))),
                    ("aggregate",
                     Apply(
                         ["action_sequence_loss", "value_loss"],
                         "loss",
                         AggregatedLoss())),
                    ("normalize",
                     Apply(
                         [("loss", "lhs")],
                         "loss",
                         mlprogram.nn.Function(mlprogram.utils.Div()),
                         constants={"rhs": 1})),
                    ("pick",
                     mlprogram.nn.Function(mlprogram.utils.Pick("loss")))
                ])),
                EvaluateSynthesizer(
                    train_dataset,
                    self.prepare_synthesizer(model, encoder, interpreter,
                                             rollout=False),
                    {}, top_n=[]),
                "generation_rate",
                metrics.transform(
                    metrics.TestCaseResult(interpreter, reference=True,
                                           metric=metrics.Iou()),
                    Threshold(0.9, dtype="float")),
                collate_fn,
                1, 1,
                Epoch(30), evaluation_interval=Epoch(30),
                snapshot_interval=Epoch(30),
                use_pretrained_model=True,
                use_pretrained_optimizer=False,
                threshold=0.9)

    @integration_test
    def test(self):
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)
        with tempfile.TemporaryDirectory() as tmpdir:
            encoder, dataset = self.pretrain(tmpdir)
            self.reinforce(dataset, encoder, tmpdir)
            result = self.evaluate(dataset, encoder, tmpdir)
        self.assertLessEqual(0.9, result.generation_rate)


if __name__ == "__main__":
    unittest.main()
