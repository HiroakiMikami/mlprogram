import torch
import torch.nn as nn
from typing import List
from mlprogram import Environment
from mlprogram.synthesizers import Synthesizer, Result
from mlprogram.languages import AST, Node, Leaf, Field
from mlprogram.languages import Token
from mlprogram.utils.data import Collate
from mlprogram.interpreters import Reference
from mlprogram.interpreters import Statement
from mlprogram.interpreters import SequentialProgram
from mlprogram.samplers \
    import SequentialProgramSampler, SamplerState, DuplicatedSamplerState


def transform_input(x):
    return Environment(inputs={"x": x})


class MockSynthesizer(Synthesizer[Environment, AST]):
    def __init__(self, asts: List[AST]):
        self.asts = asts

    def __call__(self, input: Environment, n_required_output=None):
        for i, ast in enumerate(self.asts):
            yield Result(ast, 1.0 / (i + 1), True, 1)


class MockEncoder(nn.Module):
    def forward(self, x):
        return x


class TestSequentialProgramSampler(object):
    def test_create_output(self):
        asts = [
            Node("def", [Field("name", "str", Leaf("str", "f"))]),
            Node("int", [Field("value", "int", Leaf("int", "10"))]),
            Node("float", [Field("value", "float", Leaf("float", "10.0"))]),
        ]
        sampler = SequentialProgramSampler(
            MockSynthesizer(asts),
            transform_input,
            Collate(torch.device("cpu")),
            MockEncoder(),
            to_code=lambda x: x)
        assert sampler.create_output(
            None, Environment(inputs={"code": SequentialProgram([])})
        ) is None
        assert (SequentialProgram([
                Statement(Reference("v0"), "tmp")
                ]), False) == \
            sampler.create_output(None, Environment(inputs={
                "code": SequentialProgram([
                    Statement(Reference("v0"), "tmp")
                ]),
                "unused_reference":
                    [Token(None, Reference("v0"), Reference("v0"))]
            }))
        assert (SequentialProgram([
                Statement(Reference("v1"), "tmp")
                ]), False) == \
            sampler.create_output(None, Environment(inputs={
                "code": SequentialProgram([
                    Statement(Reference("v0"), "tmp2"),
                    Statement(Reference("v1"), "tmp")
                ]),
                "unused_reference": [
                    Token(None, Reference("v0"), Reference("v0")),
                    Token(None, Reference("v1"), Reference("v1"))
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
            transform_input,
            Collate(torch.device("cpu")),
            MockEncoder(),
            to_code=lambda x: x)
        zero = SamplerState(0, sampler.initialize(0))
        samples = list(sampler.batch_k_samples([zero], [3]))
        samples.sort(key=lambda x: -x.state.score)
        assert 3 == len(samples)
        assert DuplicatedSamplerState(
            SamplerState(1, Environment(
                inputs={
                    "x": 0,
                    "unused_reference":
                        [Token("def", Reference("v0"), Reference("v0"))],
                    "code": SequentialProgram(
                        [Statement(Reference("v0"), asts[0])])
                },
                states={
                    "reference":
                        [Token("def", Reference("v0"), Reference("v0"))],
                })),
            1) == samples[0]
        assert DuplicatedSamplerState(
            SamplerState(0.5, Environment(
                inputs={
                    "x": 0,
                    "unused_reference":
                        [Token("int", Reference("v0"), Reference("v0"))],
                    "code": SequentialProgram(
                        [Statement(Reference("v0"), asts[1])])
                },
                states={
                    "reference":
                        [Token("int", Reference("v0"), Reference("v0"))],
                })),
            1) == samples[1]
        assert DuplicatedSamplerState(
            SamplerState(1.0 / 3, Environment(
                inputs={
                    "x": 0,
                    "unused_reference":
                    [Token("float", Reference("v0"), Reference("v0"))],
                    "code": SequentialProgram(
                        [Statement(Reference("v0"), asts[2])])
                },
                states={
                    "reference":
                        [Token("float", Reference("v0"), Reference("v0"))],
                })),
            1) == samples[2]

    def test_remove_used_variable(self):
        ast = Node("String", [Field("value", "str", Leaf("str", "str"))])
        asts = [
            Node("def",
                 [Field("name", "str", Leaf("String", Reference("v0")))]),
        ]
        sampler = SequentialProgramSampler(
            MockSynthesizer(asts),
            transform_input,
            Collate(torch.device("cpu")),
            MockEncoder(),
            to_code=lambda x: x)
        zero = SamplerState(0, sampler.initialize(0))
        zero.state.states["reference"] = [
            Token("str", Reference("v0"), Reference("v0"))]
        zero.state.inputs["unused_reference"] = [
            Token("str", Reference("v0"), Reference("v0"))]
        zero.state.inputs["code"] = \
            SequentialProgram([Statement(Reference("v0"), ast)])
        samples = list(sampler.batch_k_samples([zero], [1]))
        samples.sort(key=lambda x: -x.state.score)
        assert 1 == len(samples)
        assert samples[0] == DuplicatedSamplerState(
            SamplerState(1, Environment(
                inputs={
                    "x": 0,
                    "unused_reference":
                    [Token("def", Reference("v1"), Reference("v1"))],
                    "code": SequentialProgram([
                        Statement(Reference("v0"), ast),
                        Statement(Reference("v1"), asts[0])])
                },
                states={
                    "reference":
                    [Token("def", Reference("v1"), Reference("v1"))],
                })),
            1)
