import numpy as np
import unittest
from collections import OrderedDict
import logging
import sys
import tempfile
import os

import torch
import torch.optim as optim
from mlprogram.entrypoint import train_supervised, evaluate as eval
from mlprogram.entrypoint.train import Iteration
from mlprogram.entrypoint.torch import create_optimizer
from mlprogram.synthesizers import SMC
from mlprogram.samplers import ActionSequenceSampler, AstReferenceSampler
from mlprogram.encoders import ActionSequenceEncoder
from mlprogram.utils import Sequence, Map, Flatten
from mlprogram.utils.data import Collate, CollateOptions
from mlprogram.utils.transform \
    import AstToSingleActionSequence, EvaluateGroundTruth, RandomChoice
from mlprogram.nn.action_sequence import Loss, Accuracy
from mlprogram.nn import CNN2d, Apply
from mlprogram.nn.pbe_with_repl import Encoder
import mlprogram.nn.action_sequence as a_s
from mlprogram import metrics
from mlprogram.languages.csg import get_samples, IsSubtype
from mlprogram.languages.csg import Interpreter, ToAst, Dataset
from mlprogram.utils.data \
    import to_map_style_dataset, transform as data_transform
from mlprogram.utils.transform.csg import TransformCanvas
from mlprogram.utils.transform.action_sequence \
    import TransformCode, TransformGroundTruth, \
    TransformActionSequenceForRnnDecoder
from mlprogram.utils.transform.pbe_with_repl import ToEpisode

logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)


class TestCsgByPbeWithREPL(unittest.TestCase):
    def prepare_encoder(self, dataset, to_action_sequence):
        return ActionSequenceEncoder(get_samples(dataset, to_action_sequence),
                                     0)

    def prepare_model(self, encoder: ActionSequenceEncoder):
        embed = CNN2d(1, 64, 64, 2, 4, 2)
        return torch.nn.Sequential(OrderedDict([
            ("encode_input",
             Apply("input", "input_feature", embed)),
            ("encoder",
             torch.nn.Sequential(OrderedDict([
                 ("encode_variables",
                  Apply("variables", "reference_features", embed,
                        value_type="padded_tensor")),
                 ("encoder", Encoder())
             ]))),
            ("decoder",
             torch.nn.Sequential(OrderedDict([
                 ("action_sequence_reader",
                  a_s.ActionSequenceReader(encoder._rule_encoder.vocab_size,
                                           encoder._token_encoder.vocab_size,
                                           256)),
                 ("decoder",
                  a_s.RnnDecoder(2 * 64 * 1 * 1, 256, 256, 0.0)),
                 ("predictor",
                  a_s.Predictor(256, 64 * 1 * 1,
                                encoder._rule_encoder.vocab_size,
                                encoder._token_encoder.vocab_size,
                                256))
             ])))
        ]))

    def prepare_optimizer(self, model):
        return create_optimizer(optim.Adam, model)

    def prepare_synthesizer(self, model, encoder):
        collate = Collate(
            torch.device("cpu"),
            input=CollateOptions(False, 0, 0),
            variables=CollateOptions(True, 0, 0),
            previous_actions=CollateOptions(True, 0, -1),
            history=CollateOptions(False, 1, 0),
            hidden_state=CollateOptions(False, 0, 0),
            state=CollateOptions(False, 0, 0),
            ground_truth_actions=CollateOptions(True, 0, -1)
        )
        subsynthesizer = SMC(
            20, 1, 20,
            ActionSequenceSampler(
                encoder, lambda x: None, IsSubtype(),
                TransformCanvas(),
                TransformActionSequenceForRnnDecoder(encoder, train=False),
                collate, model)
        )

        sampler = AstReferenceSampler(
            subsynthesizer,
            TransformCanvas(),
            collate,
            model.encode_input)  # TODO
        return SMC(20, 1, 20, sampler, rng=np.random.RandomState(0))

    def interpreter(self):
        return Interpreter(2, 2, 8)

    def to_episode(self, encoder, interpreter, to_action_sequence):
        tcanvas = TransformCanvas()
        to_episode = ToEpisode(ToAst(), remove_used_reference=True)
        return Sequence(
            OrderedDict([
                ("choice", RandomChoice(rng=np.random.RandomState(0))),
                ("tcanvas", tcanvas),
                ("to_episode", to_episode)
            ])
        )

    def transform(self, encoder, interpreter, to_action_sequence):
        tcode = TransformCode(to_action_sequence)
        teval = TransformActionSequenceForRnnDecoder(encoder)
        tgt = TransformGroundTruth(encoder)
        return Sequence(
            OrderedDict([
                ("tcode", tcode),
                ("teval", teval),
                ("tgt", tgt)
            ])
        )

    def evaluate(self, dataset, encoder, dir):
        with tempfile.TemporaryDirectory() as tmpdir:
            interpreter = self.interpreter()
            model = self.prepare_model(encoder)
            eval(
                dir, tmpdir, dir,
                dataset, dataset,
                model,
                self.prepare_synthesizer(model, encoder),
                {"iou": metrics.TestCaseResult(interpreter, reference=True,
                                               metric=metrics.Iou())},
                (5, "iou"), top_n=[5]
            )
        results = torch.load(os.path.join(dir, "results.pt"))
        return results["valid"]

    def pretrain(self, output_dir):
        dataset = Dataset(2, 2, 1, 45, reference=True, seed=1)
        train_dataset = to_map_style_dataset(dataset, 10)

        with tempfile.TemporaryDirectory() as tmpdir:
            to_action_sequence = Sequence(OrderedDict([
                ("to_ast", ToAst()),
                ("to_sequence", AstToSingleActionSequence())
            ]))
            interpreter = self.interpreter()
            train_dataset = data_transform(
                train_dataset,
                EvaluateGroundTruth(interpreter, reference=True))
            encoder = self.prepare_encoder(dataset, to_action_sequence)

            collate = Collate(
                torch.device("cpu"),
                input=CollateOptions(False, 0, 0),
                variables=CollateOptions(True, 0, 0),
                previous_actions=CollateOptions(True, 0, -1),
                history=CollateOptions(False, 1, 0),
                hidden_state=CollateOptions(False, 0, 0),
                state=CollateOptions(False, 0, 0),
                ground_truth_actions=CollateOptions(True, 0, -1)
            )
            collate_fn = Sequence(OrderedDict([
                ("to_espisode", Map(self.to_episode(encoder,
                                                    interpreter,
                                                    to_action_sequence))),
                ("flatten", Flatten()),
                ("transform", Map(self.transform(
                    encoder, interpreter, to_action_sequence))),
                ("collate", collate)
            ]))

            model = self.prepare_model(encoder)
            optimizer = self.prepare_optimizer(model)
            train_supervised(
                tmpdir, output_dir,
                train_dataset, model, optimizer,
                Loss(), Accuracy(),
                collate_fn,
                1, Iteration(1000), interval=Iteration(100),
                num_models=1
            )
        return encoder, train_dataset

    def test(self):
        torch.manual_seed(0)
        np.random.seed(0)
        with tempfile.TemporaryDirectory() as tmpdir:
            encoder, dataset = self.pretrain(tmpdir)
            results = self.evaluate(dataset, encoder, tmpdir)
        self.assertGreaterEqual(0.9, results.metrics[5]["iou"])


if __name__ == "__main__":
    unittest.main()
