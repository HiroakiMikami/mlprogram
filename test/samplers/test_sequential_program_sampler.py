import torch
import torch.nn as nn
import unittest
from typing import List, Dict, Any
from mlprogram.synthesizers import Synthesizer, Result
from mlprogram.languages import AST, Node, Leaf, Field
from mlprogram.utils import Token
from mlprogram.utils.data import Collate
from mlprogram.interpreters import Reference
from mlprogram.interpreters import Statement
from mlprogram.interpreters import SequentialProgram
from mlprogram.samplers \
    import SequentialProgramSampler, SamplerState, DuplicatedSamplerState


class MockSynthesizer(Synthesizer[Dict[str, Any], AST]):
    def __init__(self, asts: List[AST]):
        self.asts = asts

    def __call__(self, input: Dict[str, Any], n_required_output=None):
        for i, ast in enumerate(self.asts):
            yield Result(ast, 1.0 / (i + 1), True, 1)


class MockEncoder(nn.Module):
    def forward(self, x):
        return x


class TestSequentialProgramSampler(unittest.TestCase):
    def test_create_output(self):
        asts = [
            Node("def", [Field("name", "str", Leaf("str", "f"))]),
            Node("int", [Field("value", "int", Leaf("int", "10"))]),
            Node("float", [Field("value", "float", Leaf("float", "10.0"))]),
        ]
        sampler = SequentialProgramSampler(
            MockSynthesizer(asts),
            lambda x: {"x": x},
            Collate(torch.device("cpu")),
            MockEncoder(),
            to_code=lambda x: x)
        self.assertEqual(
            None,
            sampler.create_output(None, {"code": SequentialProgram([])}))
        self.assertEqual(
            (SequentialProgram([
                Statement(Reference("v0"), "tmp")
            ]), False),
            sampler.create_output(None, {
                "code": SequentialProgram([
                    Statement(Reference("v0"), "tmp")
                ]),
                "unused_reference": [Token(None, Reference("v0"))]
            }))
        self.assertEqual(
            (SequentialProgram([
                Statement(Reference("v1"), "tmp")
            ]), False),
            sampler.create_output(None, {
                "code": SequentialProgram([
                    Statement(Reference("v0"), "tmp2"),
                    Statement(Reference("v1"), "tmp")
                ]),
                "unused_reference": [
                    Token(None, Reference("v0")),
                    Token(None, Reference("v1"))
                ]
            }))

    def test_ast_set_sample(self):
        asts = [
            Node("def", [Field("name", "str", Leaf("str", "f"))]),
            Node("int", [Field("value", "int", Leaf("int", "10"))]),
            Node("float", [Field("value", "float", Leaf("float", "10.0"))]),
        ]
        sampler = SequentialProgramSampler(
            MockSynthesizer(asts),
            lambda x: {"x": x},
            Collate(torch.device("cpu")),
            MockEncoder(),
            to_code=lambda x: x)
        zero = SamplerState(0, sampler.initialize(0))
        samples = list(sampler.batch_k_samples([zero], [3]))
        samples.sort(key=lambda x: -x.state.score)
        self.assertEqual(3, len(samples))
        self.assertEqual(
            DuplicatedSamplerState(
                SamplerState(1, {
                    "x": 0,
                    "reference": [Token("def", Reference("v0"))],
                    "unused_reference": [Token("def", Reference("v0"))],
                    "code": SequentialProgram(
                        [Statement(Reference("v0"), asts[0])])
                }),
                1),
            samples[0]
        )
        self.assertEqual(
            DuplicatedSamplerState(
                SamplerState(0.5, {
                    "x": 0,
                    "reference": [Token("int", Reference("v0"))],
                    "unused_reference": [Token("int", Reference("v0"))],
                    "code": SequentialProgram(
                        [Statement(Reference("v0"), asts[1])])
                }),
                1),
            samples[1]
        )
        self.assertEqual(
            DuplicatedSamplerState(
                SamplerState(1.0 / 3, {
                    "x": 0,
                    "reference": [Token("float", Reference("v0"))],
                    "unused_reference": [Token("float", Reference("v0"))],
                    "code": SequentialProgram(
                        [Statement(Reference("v0"), asts[2])])
                }),
                1),
            samples[2]
        )

    def test_remove_used_variable(self):
        ast = Node("String", [Field("value", "str", Leaf("str", "str"))])
        asts = [
            Node("def",
                 [Field("name", "str", Leaf("String", Reference("v0")))]),
        ]
        sampler = SequentialProgramSampler(
            MockSynthesizer(asts),
            lambda x: {"x": x},
            Collate(torch.device("cpu")),
            MockEncoder(),
            to_code=lambda x: x)
        zero = SamplerState(0, sampler.initialize(0))
        zero.state["reference"] = [Token("str", Reference("v0"))]
        zero.state["unused_reference"] = [Token("str", Reference("v0"))]
        zero.state["code"] = \
            SequentialProgram([Statement(Reference("v0"), ast)])
        samples = list(sampler.batch_k_samples([zero], [1]))
        samples.sort(key=lambda x: -x.state.score)
        self.assertEqual(1, len(samples))
        self.assertEqual(
            DuplicatedSamplerState(
                SamplerState(1, {
                    "x": 0,
                    "reference": [Token("def", Reference("v1"))],
                    "unused_reference": [Token("def", Reference("v1"))],
                    "code": SequentialProgram([
                        Statement(Reference("v0"), ast),
                        Statement(Reference("v1"), asts[0])])
                }),
                1),
            samples[0]
        )


if __name__ == "__main__":
    unittest.main()
