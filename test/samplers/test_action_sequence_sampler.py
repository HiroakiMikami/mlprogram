import torch
import torch.nn as nn
import numpy as np
from typing import List
import unittest
from mlprogram.samplers \
    import ActionSequenceSampler, SamplerState
from mlprogram.encoders import Samples, ActionSequenceEncoder
from mlprogram.languages.ast import Root
from mlprogram.actions \
    import NodeConstraint, NodeType, ExpandTreeRule
from mlprogram.utils import Token
from mlprogram.utils.data import Collate, CollateOptions
from math import log

R = NodeType(Root(), NodeConstraint.Node, False)
X = NodeType("X", NodeConstraint.Node, False)
Y = NodeType("Y", NodeConstraint.Node, False)
Y_list = NodeType("Y", NodeConstraint.Node, True)
Ysub = NodeType("Ysub", NodeConstraint.Node, False)
Str = NodeType("Str", NodeConstraint.Token, True)

Root2X = ExpandTreeRule(R, [("x", X)])
Root2Y = ExpandTreeRule(R, [("y", Y)])
X2Y_list = ExpandTreeRule(X, [("y", Y_list)])
Ysub2Str = ExpandTreeRule(Ysub, [("str", Str)])


def is_subtype(arg0, arg1):
    if arg0 == arg1:
        return True
    if arg0 == "Ysub" and arg1 == "Y":
        return True
    return False


def get_token_type(token):
    try:
        int(token)
        return "Int"
    except:  # noqa
        return "Str"


def create_encoder():
    return ActionSequenceEncoder(Samples(
        [Root2X, Root2Y, X2Y_list, Ysub2Str],
        [R, X, Y, Ysub, Y_list, Str],
        ["x", "1"]), 0)


collate = Collate(torch.device("cpu"),
                  input=CollateOptions(False, 0, -1),
                  length=CollateOptions(False, 0, -1))


def create_transform_input(reference: List[Token[str]]):
    def transform_input(kwargs):
        kwargs["reference"] = reference
        kwargs["input"] = torch.zeros((1,))
        return kwargs
    return transform_input


def transform_action_sequence(kwargs):
    kwargs["length"] = \
        torch.tensor(len(kwargs["action_sequence"].action_sequence))
    return kwargs


class EncoderModule(nn.Module):
    def forward(self, kwargs):
        return kwargs


encoder_module = EncoderModule()


class DecoderModule(nn.Module):
    def __init__(self, rule_prob, token_prob, reference_prob):
        super().__init__()
        self.rule_prob = rule_prob
        self.token_prob = token_prob
        self.reference_prob = reference_prob

    def forward(self, kwargs):
        length = kwargs["length"][0] - 1
        kwargs["rule_probs"] = self.rule_prob[length]
        kwargs["token_probs"] = self.token_prob[length]
        kwargs["reference_probs"] = self.reference_prob[length]
        return kwargs


class Module(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder


class TestActionSequenceSampler(unittest.TestCase):
    def test_initialize(self):
        sampler = ActionSequenceSampler(
            create_encoder(),
            get_token_type,
            is_subtype,
            create_transform_input([]), transform_action_sequence,
            collate,
            Module(encoder_module,
                   DecoderModule([], [], []))
        )
        s = sampler.initialize({})
        self.assertEqual(1, len(s["action_sequence"].action_sequence))
        s.pop("action_sequence")
        self.assertEqual({"input": torch.zeros((1,)), "reference": []}, s)

    def test_rule(self):
        rule_prob = torch.tensor([
            [[
                1.0,  # unknown
                1.0,  # close variadic field
                0.2,  # Root2X
                0.1,  # Root2Y
                1.0,  # X2Y_list
                1.0,  # Ysub2Str
            ]],
            [[
                1.0,  # unknown
                1.0,  # close variadic field
                1.0,  # Root2X
                1.0,  # Root2Y
                0.5,  # X2Y_list
                1.0,  # Ysub2Str
            ]]])
        token_prob = torch.tensor([[[]], [[]]])
        reference_prob = torch.tensor([[[]], [[]]])
        sampler = ActionSequenceSampler(
            create_encoder(),
            get_token_type,
            is_subtype,
            create_transform_input([]), transform_action_sequence,
            collate,
            Module(encoder_module,
                   DecoderModule(rule_prob, token_prob, reference_prob))
        )
        s = SamplerState(0.0, sampler.initialize({}))
        topk_results = list(sampler.top_k_samples([s], 1))
        self.assertEqual(1, len(topk_results))
        self.assertEqual(1, topk_results[0].state.state["length"].item())
        self.assertAlmostEqual(log(0.2), topk_results[0].state.score)
        random_results = list(sampler.batch_k_samples([s], [1]))
        self.assertEqual(1, len(random_results))
        self.assertEqual(1, random_results[0].state.state["length"].item())
        self.assertTrue(
            log(0.1) - 1e-5 <= random_results[0].state.score
            <= log(0.2) + 1e-5)

        next = list(sampler.top_k_samples(
            [s.state for s in topk_results], 1))[0]
        self.assertEqual(2, next.state.state["length"].item())
        self.assertAlmostEqual(log(0.2) + log(0.5), next.state.score)

    def test_variadic_rule(self):
        rule_prob = torch.tensor([
            [[
                1.0,  # unknown
                1.0,  # close variadic field
                0.2,  # Root2X
                0.1,  # Root2Y
                1.0,  # X2Y_list
                1.0,  # Ysub2Str
            ]],
            [[
                1.0,  # unknown
                1.0,  # close variadic field
                1.0,  # Root2X
                1.0,  # Root2Y
                0.5,  # X2Y_list
                1.0,  # Ysub2Str
            ]],
            [[
                1.0,  # unknown
                0.8,  # close variadic field
                1.0,  # Root2X
                1.0,  # Root2Y
                1.0,  # X2Y_list
                0.2,  # Ysub2Str
            ]]])
        token_prob = torch.tensor([[[]], [[]], [[]]])
        reference_prob = torch.tensor([[[]], [[]], [[]]])
        sampler = ActionSequenceSampler(
            create_encoder(),
            get_token_type,
            is_subtype,
            create_transform_input([]), transform_action_sequence,
            collate,
            Module(encoder_module,
                   DecoderModule(rule_prob, token_prob, reference_prob))
        )
        s = SamplerState(0.0, sampler.initialize({}))
        results = [s.state for s in sampler.top_k_samples([s], 1)]
        results = [s.state for s in sampler.top_k_samples(results, 1)]
        topk_results = \
            list(sampler.top_k_samples(results, 2))
        self.assertEqual(2, len(topk_results))
        self.assertEqual(3, topk_results[0].state.state["length"].item())
        self.assertAlmostEqual(log(0.2) + log(0.5) +
                               log(0.8), topk_results[0].state.score)
        self.assertAlmostEqual(log(0.2) + log(0.5) +
                               log(0.2), topk_results[1].state.score)
        random_results = list(sampler.batch_k_samples(results[:1], [1]))
        self.assertEqual(1, len(random_results))
        self.assertEqual(3, random_results[0].state.state["length"].item())
        self.assertTrue(
            (log(0.2) + log(0.5) + log(0.2) - 1e-5 <=
                random_results[0].state.score <=
                log(0.2) + log(0.5) + log(0.8) + 1e-5))

    def test_token(self):
        rule_prob = torch.tensor([
            [[
                1.0,  # unknown
                1.0,  # close variadic field
                0.1,  # Root2X
                0.2,  # Root2Y
                1.0,  # X2Y_list
                1.0,  # Ysub2Str
            ]],
            [[
                1.0,  # unknown
                1.0,  # close variadic field
                1.0,  # Root2X
                1.0,  # Root2Y
                1.0,  # X2Y_list
                1.0,  # Ysub2Str
            ]],
            [[0.0,
              0.8,  # CloseVariadicField
              0.0, 0.0, 0.0, 0.0]]])
        token_prob = torch.tensor([
            [[0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0]],
            [[
                1.0,  # Unknown
                0.2,  # x
                0.8,  # 1
            ]]])
        reference_prob = torch.tensor([[[]], [[]], [[]]])
        sampler = ActionSequenceSampler(
            create_encoder(),
            get_token_type,
            is_subtype,
            create_transform_input([]), transform_action_sequence,
            collate,
            Module(encoder_module,
                   DecoderModule(rule_prob, token_prob, reference_prob))
        )
        s = SamplerState(0.0, sampler.initialize({}))
        results = [s.state for s in sampler.top_k_samples([s], 1)]
        results = [s.state for s in sampler.top_k_samples(results, 1)]
        topk_results = list(sampler.top_k_samples(results, 2))
        self.assertEqual(2, len(topk_results))
        self.assertEqual(3, topk_results[0].state.state["length"].item())
        self.assertAlmostEqual(log(0.2) + log(0.8),
                               topk_results[0].state.score)
        self.assertAlmostEqual(log(0.2) + log(0.2),
                               topk_results[1].state.score)
        random_results = list(sampler.batch_k_samples(results[:1], [1]))
        self.assertEqual(1, len(random_results))
        self.assertEqual(3, random_results[0].state.state["length"].item())
        self.assertTrue(
            (log(0.2) + log(0.2) - 1e-5 <=
                random_results[0].state.score <=
                log(0.2) + log(0.8) + 1e-5))

    def test_reference(self):
        torch.manual_seed(0)
        rule_prob = torch.tensor([
            [[
                1.0,  # unknown
                1.0,  # close variadic field
                0.1,  # Root2X
                0.2,  # Root2Y
                1.0,  # X2Y_list
                1.0,  # Ysub2Str
            ]],
            [[
                1.0,  # unknown
                1.0,  # close variadic field
                1.0,  # Root2X
                1.0,  # Root2Y
                1.0,  # X2Y_list
                1.0,  # Ysub2Str
            ]],
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
        token_prob = torch.tensor([
            [[0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0]],
            [[
                1.0,  # Unknown
                0.8,  # x
                0.2,  # 1
            ]]])
        reference_prob = \
            torch.tensor([[[0.0, 0.0]], [[0.0, 0.0]], [[0.1, 0.1]]])
        sampler = ActionSequenceSampler(
            create_encoder(),
            get_token_type,
            is_subtype,
            create_transform_input([Token("Str", "x"), Token("Str", "x")]),
            transform_action_sequence,
            collate,
            Module(encoder_module,
                   DecoderModule(rule_prob, token_prob, reference_prob)),
            rng=np.random.RandomState(0)
        )
        s = SamplerState(0.0, sampler.initialize({}))
        results = [s.state for s in sampler.top_k_samples([s], 1)]
        results = [s.state for s in sampler.top_k_samples(results, 1)]
        topk_results = list(sampler.top_k_samples(results, 1))
        self.assertEqual(1, len(topk_results))
        self.assertEqual(3, topk_results[0].state.state["length"].item())
        self.assertAlmostEqual(log(0.2) + log(1.),
                               topk_results[0].state.score)
        random_results = list(sampler.batch_k_samples(results[:1], [1]))
        self.assertEqual(1, len(random_results))
        self.assertEqual(3, random_results[0].state.state["length"].item())
        self.assertAlmostEqual(log(0.2) + log(1.0),
                               random_results[0].state.score)


if __name__ == "__main__":
    unittest.main()
