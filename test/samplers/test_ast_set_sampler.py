import torch
import torch.nn as nn
import unittest
from typing import List, Dict, Any
from mlprogram.synthesizers import Synthesizer, Result
from mlprogram.asts import AST, Node, Leaf, Field
from mlprogram.utils import Token
from mlprogram.utils.data import Collate
from mlprogram.samplers import AstSetSampler, SamplerState, Reference


class MockSynthesizer(Synthesizer[Dict[str, Any], AST]):
    def __init__(self, asts: List[AST]):
        self.asts = asts

    def __call__(self, input: Dict[str, Any]):
        for i, ast in enumerate(self.asts):
            yield Result(ast, 1.0 / (i + 1))


class MockEncoder(nn.Module):
    def forward(self, x):
        return x


class TestAstSetSampler(unittest.TestCase):
    def test_ast_set_sample(self):
        asts = [
            Node("def", [Field("name", "str", Leaf("str", "f"))]),
            Node("int", [Field("value", "int", Leaf("int", "10"))]),
            Node("float", [Field("value", "float", Leaf("float", "10.0"))]),
        ]
        sampler = AstSetSampler(
            MockSynthesizer(asts),
            lambda x: {"x": x},
            Collate(torch.device("cpu")),
            MockEncoder())
        zero = SamplerState(0, sampler.initialize(0))
        samples = list(sampler.k_samples([zero], 3))
        samples.sort(key=lambda x: -x.score)
        self.assertEqual(3, len(samples))
        self.assertEqual(
            SamplerState(1, {
                "x": 0,
                "reference": [Token("def", Reference("v0"))],
                "mapping": [(Reference("v0"), asts[0])]
            }),
            samples[0]
        )
        self.assertEqual(
            SamplerState(0.5, {
                "x": 0,
                "reference": [Token("int", Reference("v0"))],
                "mapping": [(Reference("v0"), asts[1])]
            }),
            samples[1]
        )
        self.assertEqual(
            SamplerState(1.0 / 3, {
                "x": 0,
                "reference": [Token("float", Reference("v0"))],
                "mapping": [(Reference("v0"), asts[2])]
            }),
            samples[2]
        )

    def test_remove_used_variable(self):
        ast = Node("String", [Field("value", "str", Leaf("str", "str"))])
        asts = [
            Node("def",
                 [Field("name", "str", Leaf("String", Reference("v0")))]),
        ]
        sampler = AstSetSampler(
            MockSynthesizer(asts),
            lambda x: {"x": x},
            Collate(torch.device("cpu")),
            MockEncoder())
        zero = SamplerState(0, sampler.initialize(0))
        zero.state["reference"] = [Token("str", Reference("v0"))]
        zero.state["mapping"] = [(Reference("v0"), ast)]
        samples = list(sampler.k_samples([zero], 1))
        samples.sort(key=lambda x: -x.score)
        self.assertEqual(1, len(samples))
        self.assertEqual(
            SamplerState(1, {
                "x": 0,
                "reference": [Token("def", Reference("v1"))],
                "mapping": [(Reference("v0"), ast), (Reference("v1"), asts[0])]
            }),
            samples[0]
        )


if __name__ == "__main__":
    unittest.main()
