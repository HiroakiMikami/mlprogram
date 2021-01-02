import logging
import os
import random
import sys
import tempfile
from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim

import mlprogram.nn
import mlprogram.nn.action_sequence as a_s
import mlprogram.samplers
from mlprogram import metrics
from mlprogram.builtins import Apply, Div, Flatten, Mul, Pick, Threshold
from mlprogram.encoders import ActionSequenceEncoder
from mlprogram.entrypoint import EvaluateSynthesizer
from mlprogram.entrypoint import evaluate as eval
from mlprogram.entrypoint import train_REINFORCE, train_supervised
from mlprogram.entrypoint.modules.torch import Optimizer, Reshape
from mlprogram.entrypoint.train import Epoch
from mlprogram.functools import Compose, Map, Sequence
from mlprogram.languages.csg import (
    Dataset,
    Expander,
    Interpreter,
    IsSubtype,
    Parser,
    get_samples,
)
from mlprogram.languages.csg.transforms import (
    AddTestCases,
    TransformInputs,
    TransformVariables,
)
from mlprogram.nn import MLP, AggregatedLoss, CNN2d
from mlprogram.nn.action_sequence import Loss
from mlprogram.nn.pbe_with_repl import Encoder
from mlprogram.samplers import (
    ActionSequenceSampler,
    FilteredSampler,
    SamplerWithValueNetwork,
    SequentialProgramSampler,
)
from mlprogram.synthesizers import SMC, FilteredSynthesizer, SynthesizerWithTimeout
from mlprogram.transforms.action_sequence import (
    AddPreviousActions,
    AddState,
    EncodeActionSequence,
    GroundTruthToActionSequence,
)
from mlprogram.transforms.pbe import ToEpisode
from mlprogram.utils.data import Collate, CollateOptions, to_map_style_dataset
from mlprogram.utils.data import transform as data_transform

logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)


class TestCsgByPbeWithREPL(object):
    def prepare_encoder(self, dataset, parser):
        return ActionSequenceEncoder(get_samples(dataset, parser, reference=True),
                                     0)

    def prepare_model(self, encoder: ActionSequenceEncoder):
        return torch.nn.Sequential(OrderedDict([
            ("encode_input",
             Apply([("test_case_tensor", "x")],
                   "test_case_feature",
                   CNN2d(1, 16, 32, 2, 2, 2))),
            ("encoder",
             Apply(
                 module=Encoder(CNN2d(2, 16, 32, 2, 2, 2)),
                 in_keys=["test_case_tensor",
                          "variables_tensor", "test_case_feature"],
                 out_key=["reference_features", "input_feature"]
             )),
            ("decoder",
             torch.nn.Sequential(OrderedDict([
                 ("action_embedding",
                  Apply(
                      module=a_s.PreviousActionsEmbedding(
                          n_rule=encoder._rule_encoder.vocab_size,
                          n_token=encoder._token_encoder.vocab_size,
                          embedding_size=256,
                      ),
                      in_keys=["previous_actions"],
                      out_key="action_features"
                  )),
                 ("decoder",
                  Apply(
                      module=a_s.LSTMDecoder(
                          inject_input=a_s.CatInput(),
                          input_feature_size=2 * 16 * 8 * 8,
                          action_feature_size=256,
                          output_feature_size=512,
                          dropout=0.0
                      ),
                      in_keys=["input_feature", "action_features", "hidden_state",
                               "state"],
                      out_key=["action_features", "hidden_state", "state"]
                  )),
                 ("predictor",
                  Apply(
                      module=a_s.Predictor(512, 16 * 8 * 8,
                                           encoder._rule_encoder.vocab_size,
                                           encoder._token_encoder.vocab_size,
                                           512),
                      in_keys=["action_features", "reference_features"],
                      out_key=["rule_probs", "token_probs", "reference_probs"]))
             ]))),
            ("value",
             Apply([("input_feature", "x")], "value",
                   MLP(16 * 8 * 8 * 2, 1, 512, 2,
                       activation=torch.nn.Sigmoid()),
                   ))
        ]))

    def prepare_optimizer(self, model):
        return Optimizer(optim.Adam, model)

    def prepare_synthesizer(self, model, encoder, interpreter, rollout=True):
        collate = Collate(
            torch.device("cpu"),
            test_case_tensor=CollateOptions(False, 0, 0),
            input_feature=CollateOptions(False, 0, 0),
            test_case_feature=CollateOptions(False, 0, 0),
            reference_features=CollateOptions(True, 0, 0),
            variables_tensor=CollateOptions(True, 0, 0),
            previous_actions=CollateOptions(True, 0, -1),
            hidden_state=CollateOptions(False, 0, 0),
            state=CollateOptions(False, 0, 0),
            ground_truth_actions=CollateOptions(True, 0, -1)
        )
        subsampler = ActionSequenceSampler(
            encoder, IsSubtype(),
            Sequence(OrderedDict([
                ("tinput",
                 Apply(
                     module=TransformInputs(),
                     in_keys=["test_cases"],
                     out_key="test_case_tensor",
                 )),
                ("tvariable",
                 Apply(
                     module=TransformVariables(),
                     in_keys=["variables", "test_case_tensor"],
                     out_key="variables_tensor"
                 )),
            ])),
            Compose(OrderedDict([
                ("add_previous_actions",
                 Apply(
                    module=AddPreviousActions(encoder, n_dependent=1),
                    in_keys=["action_sequence", "reference"],
                    out_key="previous_actions",
                    constants={"train": False},
                    )),
                ("add_state", AddState("state")),
                ("add_hidden_state", AddState("hidden_state"))
            ])),
            collate, model,
            rng=np.random.RandomState(0))
        subsampler = mlprogram.samplers.transform(
            subsampler,
            Parser().unparse
        )
        subsynthesizer = SMC(
            5, 1,
            subsampler,
            max_try_num=1,
            to_key=Pick("action_sequence"),
            rng=np.random.RandomState(0)
        )

        sampler = SequentialProgramSampler(
            subsynthesizer,
            Apply(
                module=TransformInputs(),
                in_keys=["test_cases"],
                out_key="test_case_tensor",
            ),
            collate,
            model.encode_input,
            interpreter=interpreter,
            expander=Expander(),
            rng=np.random.RandomState(0))
        if rollout:
            sampler = FilteredSampler(
                sampler,
                metrics.use_environment(
                    metric=metrics.TestCaseResult(
                        interpreter,
                        metric=metrics.use_environment(
                            metric=metrics.Iou(),
                            in_keys=["actual", "expected"],
                            value_key="actual",
                        )
                    ),
                    in_keys=["test_cases", "actual"],
                    value_key="actual"
                ),
                0.9
            )
            return SMC(4, 20, sampler, rng=np.random.RandomState(0),
                       to_key=Pick("interpreter_state"), max_try_num=1)
        else:
            sampler = SamplerWithValueNetwork(
                sampler,
                Sequence(OrderedDict([
                    ("tinput",
                     Apply(
                         module=TransformInputs(),
                         in_keys=["test_cases"],
                         out_key="test_case_tensor",
                     )),
                    ("tvariable",
                     Apply(
                         module=TransformVariables(),
                         in_keys=["variables", "test_case_tensor"],
                         out_key="variables_tensor"
                     )),
                ])),
                collate,
                torch.nn.Sequential(OrderedDict([
                    ("encoder", model.encoder),
                    ("value", model.value),
                    ("pick",
                     mlprogram.nn.Function(
                         Pick("value")))
                ])))

            synthesizer = SynthesizerWithTimeout(
                SMC(4, 20, sampler, rng=np.random.RandomState(0),
                    to_key=Pick("interpreter_state")),
                1
            )
            return FilteredSynthesizer(
                synthesizer,
                metrics.use_environment(
                    metric=metrics.TestCaseResult(
                        interpreter,
                        metric=metrics.use_environment(
                            metric=metrics.Iou(),
                            in_keys=["actual", "expected"],
                            value_key="actual",
                        )
                    ),
                    in_keys=["test_cases", "actual"],
                    value_key="actual"
                ),
                0.9
            )

    def interpreter(self):
        return Interpreter(2, 2, 8, delete_used_reference=True)

    def to_episode(self, encoder, interpreter):
        return ToEpisode(interpreter, Expander())

    def transform(self, encoder, interpreter, parser):
        tcode = Apply(
            module=GroundTruthToActionSequence(parser),
            in_keys=["ground_truth"],
            out_key="action_sequence"
        )
        aaction = Apply(
            module=AddPreviousActions(encoder, n_dependent=1),
            in_keys=["action_sequence", "reference"],
            constants={"train": True},
            out_key="previous_actions",
        )
        tgt = Apply(
            module=EncodeActionSequence(encoder),
            in_keys=["action_sequence", "reference"],
            out_key="ground_truth_actions",
        )
        return Sequence(
            OrderedDict([
                ("tinput",
                 Apply(
                     module=TransformInputs(),
                     in_keys=["test_cases"],
                     out_key="test_case_tensor",
                 )),
                ("tvariable",
                 Apply(
                     module=TransformVariables(),
                     in_keys=["variables", "test_case_tensor"],
                     out_key="variables_tensor"
                 )),
                ("tcode", tcode),
                ("aaction", aaction),
                ("add_state", AddState("state")),
                ("add_hidden_state", AddState("hidden_state")),
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
            )
        return torch.load(os.path.join(dir, "result.pt"))

    def pretrain(self, output_dir):
        dataset = Dataset(2, 1, 2, 1, 45, seed=1)
        train_dataset = to_map_style_dataset(dataset, 10)

        with tempfile.TemporaryDirectory() as tmpdir:
            interpreter = self.interpreter()
            train_dataset = data_transform(
                train_dataset,
                Apply(
                    module=AddTestCases(interpreter),
                    in_keys=["ground_truth"],
                    out_key="test_cases",
                    is_out_supervision=False,
                ))
            encoder = self.prepare_encoder(dataset, Parser())

            collate = Collate(
                torch.device("cpu"),
                test_case_tensor=CollateOptions(False, 0, 0),
                variables_tensor=CollateOptions(True, 0, 0),
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
                ("collate", collate.collate)
            ]))

            model = self.prepare_model(encoder)
            optimizer = self.prepare_optimizer(model)
            train_supervised(
                tmpdir, output_dir,
                train_dataset, model, optimizer,
                torch.nn.Sequential(OrderedDict([
                    ("loss",
                     Apply(
                         module=Loss(
                             reduction="sum",
                         ),
                         in_keys=[
                             "rule_probs",
                             "token_probs",
                             "reference_probs",
                             "ground_truth_actions",
                         ],
                         out_key="action_sequence_loss",
                     )),
                    ("normalize",  # divided by batch_size
                     Apply(
                         [("action_sequence_loss", "lhs")],
                         "loss",
                         mlprogram.nn.Function(Div()),
                         constants={"rhs": 1})),
                    ("pick",
                     mlprogram.nn.Function(
                         Pick("loss")))
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
                test_case_tensor=CollateOptions(False, 0, 0),
                variables_tensor=CollateOptions(True, 0, 0),
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
                ("collate", collate.collate)
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
                         ("loss",
                          Apply(
                              module=mlprogram.nn.action_sequence.Loss(
                                  reduction="none"),
                              in_keys=[
                                  "rule_probs",
                                  "token_probs",
                                  "reference_probs",
                                  "ground_truth_actions",
                              ],
                              out_key="action_sequence_loss",
                          )),
                         ("weight_by_reward",
                             Apply(
                                 [("reward", "lhs"),
                                  ("action_sequence_loss", "rhs")],
                                 "action_sequence_loss",
                                 mlprogram.nn.Function(Mul())))
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
                    ("reweight",
                     Apply(
                         [("value_loss", "lhs")],
                         "value_loss",
                         mlprogram.nn.Function(Mul()),
                         constants={"rhs": 1e-2})),
                    ("aggregate",
                     Apply(
                         ["action_sequence_loss", "value_loss"],
                         "loss",
                         AggregatedLoss())),
                    ("normalize",
                     Apply(
                         [("loss", "lhs")],
                         "loss",
                         mlprogram.nn.Function(Div()),
                         constants={"rhs": 1})),
                    ("pick",
                     mlprogram.nn.Function(
                         Pick("loss")))
                ])),
                EvaluateSynthesizer(
                    train_dataset,
                    self.prepare_synthesizer(model, encoder, interpreter,
                                             rollout=False),
                    {}, top_n=[]),
                "generation_rate",
                metrics.use_environment(
                    metric=metrics.TestCaseResult(
                        interpreter=interpreter,
                        metric=metrics.use_environment(
                            metric=metrics.Iou(),
                            in_keys=["actual", "expected"],
                            value_key="actual",
                        )
                    ),
                    in_keys=["test_cases", "actual"],
                    value_key="actual",
                    transform=Threshold(threshold=0.9, dtype="float"),
                ),
                collate_fn,
                1, 1,
                Epoch(10), evaluation_interval=Epoch(10),
                snapshot_interval=Epoch(10),
                use_pretrained_model=True,
                use_pretrained_optimizer=False,
                threshold=1.0)

    def test(self):
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)
        with tempfile.TemporaryDirectory() as tmpdir:
            encoder, dataset = self.pretrain(tmpdir)
            self.reinforce(dataset, encoder, tmpdir)
            result = self.evaluate(dataset, encoder, tmpdir)
        assert 1.0 <= result.generation_rate
