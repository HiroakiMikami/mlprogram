import torch
import torch.nn as nn
import unittest
from mlprogram.samplers \
    import ActionSequenceSampler, ActionSequenceSamplerInput, SamplerState, \
    Token
from mlprogram.encoders import Samples, ActionSequenceEncoder
from mlprogram.asts import Root
from mlprogram.actions \
    import NodeConstraint, NodeType, ActionOptions, ExpandTreeRule, CloseNode
from mlprogram.utils.data import Collate, CollateOptions
from math import log

R = NodeType(Root(), NodeConstraint.Node)
X = NodeType("X", NodeConstraint.Node)
Y = NodeType("Y", NodeConstraint.Node)
Y_list = NodeType("Y", NodeConstraint.Variadic)
Ysub = NodeType("Ysub", NodeConstraint.Node)
Str = NodeType("Str", NodeConstraint.Token)

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
    if token == CloseNode():
        return None
    try:
        int(token)
        return "Int"
    except:  # noqa
        return "Str"


def create_encoder(options: ActionOptions):
    return ActionSequenceEncoder(Samples([Root2X, Root2Y, X2Y_list, Ysub2Str],
                                         [R, X, Y, Ysub, Y_list, Str],
                                         ["x", "1"], options), 0)


collate = Collate(torch.device("cpu"), length=CollateOptions(False, 0, -1))


class DecoderModule(nn.Module):
    def __init__(self, rule_prob, token_prob, copy_prob):
        super().__init__()
        self.rule_prob = rule_prob
        self.token_prob = token_prob
        self.copy_prob = copy_prob

    def forward(self, **kwargs):
        length = kwargs["length"][0] - 1
        kwargs["rule_probs"] = self.rule_prob[length]
        kwargs["token_probs"] = self.token_prob[length]
        kwargs["copy_probs"] = self.copy_prob[length]
        return kwargs


class TestActionSequenceSampler(unittest.TestCase):
    def test_initialize(self):
        sampler = ActionSequenceSampler(
            create_encoder(ActionOptions(True, True)),
            get_token_type,
            is_subtype,
            lambda x, y: {"length": len(x.sequence)},
            collate,
            DecoderModule([], [], [])
        )
        s = sampler.initialize(ActionSequenceSamplerInput({}, []))
        self.assertEqual({}, s.state)
        self.assertEqual([], s.reference)
        self.assertEqual(1, len(s.action_sequence._action_sequence))

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
        copy_prob = torch.tensor([[[]], [[]]])
        sampler = ActionSequenceSampler(
            create_encoder(ActionOptions(True, True)),
            get_token_type,
            is_subtype,
            lambda x, y: {"length": torch.tensor([len(x._action_sequence)])},
            collate,
            DecoderModule(rule_prob, token_prob, copy_prob)
        )
        s = SamplerState(0.0,
                         sampler.initialize(ActionSequenceSamplerInput({}, []))
                         )
        topk_results = list(sampler.top_k_samples([s], 1))
        self.assertEqual(1, len(topk_results))
        self.assertEqual(1, topk_results[0].state.state["length"].item())
        self.assertAlmostEqual(log(0.2), topk_results[0].score)
        random_results = list(sampler.random_samples(s, 1))
        self.assertEqual(1, len(random_results))
        self.assertEqual(1, random_results[0].state.state["length"].item())
        self.assertTrue(
            log(0.1) - 1e-5 <= random_results[0].score <= log(0.2) + 1e-5)

        next = list(sampler.top_k_samples(topk_results, 1))[0]
        self.assertEqual(2, next.state.state["length"].item())
        self.assertAlmostEqual(log(0.2) + log(0.5), next.score)

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
        copy_prob = torch.tensor([[[]], [[]], [[]]])
        sampler = ActionSequenceSampler(
            create_encoder(ActionOptions(True, True)),
            get_token_type,
            is_subtype,
            lambda x, y: {"length": torch.tensor([len(x._action_sequence)])},
            collate,
            DecoderModule(rule_prob, token_prob, copy_prob)
        )
        s = SamplerState(0.0,
                         sampler.initialize(ActionSequenceSamplerInput({}, []))
                         )
        results = list(sampler.top_k_samples([s], 1))
        results = list(sampler.top_k_samples(results, 1))
        topk_results = list(sampler.top_k_samples(results, 2))
        self.assertEqual(2, len(topk_results))
        self.assertEqual(3, topk_results[0].state.state["length"].item())
        self.assertAlmostEqual(log(0.2) + log(0.5) +
                               log(0.8), topk_results[0].score)
        self.assertAlmostEqual(log(0.2) + log(0.5) +
                               log(0.2), topk_results[1].score)
        random_results = list(sampler.random_samples(results[0], 1))
        self.assertEqual(1, len(random_results))
        self.assertEqual(3, random_results[0].state.state["length"].item())
        self.assertTrue(
            (log(0.2) + log(0.5) + log(0.2) - 1e-5 <=
                random_results[0].score <=
                log(0.2) + log(0.5) + log(0.8) + 1e-5))

    def test_variadic_rule_not_retain_variadic_fields(self):
        rule_prob = torch.tensor([
            [[
                1.0,  # unknown
                0.2,  # Root2X
                0.1,  # Root2Y
                1.0,  # X2Y_list
                1.0,  # Ysub2Str
            ]],
            [[
                1.0,  # unknown
                1.0,  # Root2X
                1.0,  # Root2Y
                0.5,  # X2Y_list
                1.0,  # Ysub2Str
            ]],
            [[
                1.0,  # unknownx
                1.0,  # Root2X
                1.0,  # Root2Y
                1.0,  # X2Y_list
                0.2,  # Ysub2Str
            ]]])
        token_prob = torch.tensor([[[]], [[]], [[]]])
        copy_prob = torch.tensor([[[]], [[]], [[]]])
        sampler = ActionSequenceSampler(
            create_encoder(ActionOptions(False, True)),
            get_token_type,
            is_subtype,
            lambda x, y: {"length": torch.tensor([len(x._action_sequence)])},
            collate,
            DecoderModule(rule_prob, token_prob, copy_prob),
            options=ActionOptions(False, True)
        )
        s = SamplerState(0.0,
                         sampler.initialize(ActionSequenceSamplerInput({}, []))
                         )
        results = list(sampler.top_k_samples([s], 1))
        results = list(sampler.top_k_samples(results, 1))
        topk_results = list(sampler.top_k_samples(results, 1))[0]
        self.assertEqual(3, topk_results.state.state["length"].item())
        self.assertAlmostEqual(log(0.2) + log(0.5) + log(0.2),
                               topk_results.score)
        random_results = list(sampler.random_samples(results[0], 1))[0]
        self.assertEqual(3, random_results.state.state["length"].item())
        self.assertAlmostEqual(log(0.2) + log(0.5) + log(0.2),
                               random_results.score)

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
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
        token_prob = torch.tensor([
            [[0.0, 0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0, 0.0]],
            [[
                1.0,  # Unknown
                0.8,  # CloseNode
                0.2,  # x
                0.8,  # 1
            ]]])
        copy_prob = torch.tensor([[[]], [[]], [[]]])
        sampler = ActionSequenceSampler(
            create_encoder(ActionOptions(True, True)),
            get_token_type,
            is_subtype,
            lambda x, y: {"length": torch.tensor([len(x._action_sequence)])},
            collate,
            DecoderModule(rule_prob, token_prob, copy_prob)
        )
        s = SamplerState(0.0,
                         sampler.initialize(ActionSequenceSamplerInput({}, []))
                         )
        results = list(sampler.top_k_samples([s], 1))
        results = list(sampler.top_k_samples(results, 1))
        topk_results = list(sampler.top_k_samples(results, 2))
        self.assertEqual(2, len(topk_results))
        self.assertEqual(3, topk_results[0].state.state["length"].item())
        self.assertAlmostEqual(log(0.2) + log(0.8),
                               topk_results[0].score)
        self.assertAlmostEqual(log(0.2) + log(0.2),
                               topk_results[1].score)
        random_results = list(sampler.random_samples(results[0], 1))
        self.assertEqual(1, len(random_results))
        self.assertEqual(3, random_results[0].state.state["length"].item())
        self.assertTrue(
            (log(0.2) + log(0.2) - 1e-5 <=
                random_results[0].score <=
                log(0.2) + log(0.8) + 1e-5))

    def test_notsplittedd_token(self):
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
                0.2,  # x
                0.8,  # 1
            ]]])
        copy_prob = torch.tensor([[[]], [[]], [[]]])
        sampler = ActionSequenceSampler(
            create_encoder(ActionOptions(True, False)),
            get_token_type,
            is_subtype,
            lambda x, y: {"length": torch.tensor([len(x._action_sequence)])},
            collate,
            DecoderModule(rule_prob, token_prob, copy_prob),
            options=ActionOptions(True, False)
        )
        s = SamplerState(0.0,
                         sampler.initialize(ActionSequenceSamplerInput({}, []))
                         )
        results = list(sampler.top_k_samples([s], 1))
        results = list(sampler.top_k_samples(results, 1))
        topk_results = list(sampler.top_k_samples(results, 2))
        self.assertEqual(1, len(topk_results))
        self.assertEqual(3, topk_results[0].state.state["length"].item())
        self.assertAlmostEqual(log(0.2) + log(0.2),
                               topk_results[0].score)
        random_results = list(sampler.random_samples(results[0], 1))
        self.assertEqual(1, len(random_results))
        self.assertEqual(3, random_results[0].state.state["length"].item())
        self.assertAlmostEqual(log(0.2) + log(0.2), random_results[0].score)

    def test_copy(self):
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
            [[0.0, 0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0, 0.0]],
            [[
                1.0,  # Unknown
                0.8,  # CloseNode
                0.2,  # x
                0.8,  # 1
            ]]])
        copy_prob = torch.tensor([[[0.0]], [[0.0]], [[0.8]]])
        sampler = ActionSequenceSampler(
            create_encoder(ActionOptions(True, True)),
            get_token_type,
            is_subtype,
            lambda x, y: {"length": torch.tensor([len(x._action_sequence)])},
            collate,
            DecoderModule(rule_prob, token_prob, copy_prob)
        )
        s = SamplerState(0.0,
                         sampler.initialize(ActionSequenceSamplerInput(
                             {},
                             [Token("Str", "x")]))
                         )
        results = list(sampler.top_k_samples([s], 1))
        results = list(sampler.top_k_samples(results, 1))
        topk_results = list(sampler.top_k_samples(results, 1))
        self.assertEqual(1, len(topk_results))
        self.assertEqual(3, topk_results[0].state.state["length"].item())
        self.assertAlmostEqual(log(0.2) + log(1.0),
                               topk_results[0].score)
        random_results = list(sampler.random_samples(results[0], 1))
        self.assertEqual(1, len(random_results))
        self.assertEqual(3, random_results[0].state.state["length"].item())
        print(random_results[0].state.action_sequence._action_sequence)
        self.assertAlmostEqual(log(0.2) + log(1.0),
                               random_results[0].score)


if __name__ == "__main__":
    unittest.main()
