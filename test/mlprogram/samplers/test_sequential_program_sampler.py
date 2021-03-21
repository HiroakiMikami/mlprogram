from typing import List

import torch.nn as nn

from mlprogram.builtins import Environment
from mlprogram.languages import BatchedState, Expander, Interpreter, Token
from mlprogram.samplers import (
    DuplicatedSamplerState,
    SamplerState,
    SequentialProgramSampler,
)
from mlprogram.synthesizers import Result, Synthesizer
from mlprogram.utils.data import Collate


def transform_input(x):
    return Environment({"test_cases": x})


class MockSynthesizer(Synthesizer[Environment, str]):
    def __init__(self, asts: List[str]):
        self.asts = asts

    def __call__(self, input: Environment, n_required_output=None):
        for i, ast in enumerate(self.asts):
            yield Result(ast, 1.0 / (i + 1), True, 1)


class MockInterpreter(Interpreter[str, str, str, str, str]):
    def eval(self, code, inputs):
        return ["#" + code for _ in inputs]

    def create_state(self, inputs):
        return BatchedState({}, {}, [], inputs)

    def execute(self, code, state):
        output = self.eval(code, state.context)
        state = state.clone()
        state.environment[code] = output
        state.type_environment[code] = None
        state.history.append(code)
        state.context = output
        return state


class MockEncoder(nn.Module):
    def forward(self, x):
        return x


class MockExpander(Expander[int]):
    def expand(self, code):
        return [code]

    def unexpand(self, code):
        return "\n".join(code)


class TestSequentialProgramSampler(object):
    def test_create_output(self):
        sampler = SequentialProgramSampler(
            MockSynthesizer([]),
            transform_input,
            Collate(),
            MockEncoder(),
            MockExpander(),
            MockInterpreter())
        assert sampler.create_output(
            None,
            Environment({"interpreter_state": BatchedState({}, {}, [], [None])})
        ) is None
        assert sampler.create_output(None, Environment({
            "interpreter_state": BatchedState({}, {}, ["tmp"], [None])
        })) == ("tmp", False)
        assert sampler.create_output(None, Environment({
            "interpreter_state": BatchedState({}, {}, ["line0", "line1"], [None])
        })) == ("line0\nline1", False)

    def test_ast_set_sample(self):
        asts = ["c0", "c1", "c2"]
        sampler = SequentialProgramSampler(
            MockSynthesizer(asts),
            transform_input,
            Collate(),
            MockEncoder(),
            MockExpander(),
            MockInterpreter())
        zero = SamplerState(0, sampler.initialize([(None, None)]))
        samples = list(sampler.batch_k_samples([zero], [3]))
        samples.sort(key=lambda x: -x.state.score)
        assert 3 == len(samples)
        assert samples[0] == DuplicatedSamplerState(
            SamplerState(1, Environment(
                {
                    "test_cases": [(None, None)],
                    "reference": [Token(None, str(asts[0]), str(asts[0]))],
                    "variables": [["#" + str(asts[0])]],
                    "interpreter_state": BatchedState(
                        {str(asts[0]): None},
                        {str(asts[0]): ["#" + str(asts[0])]},
                        [str(asts[0])],
                        ["#" + str(asts[0])]
                    )
                })),
            1)
        assert DuplicatedSamplerState(
            SamplerState(0.5, Environment(
                {
                    "test_cases": [(None, None)],
                    "reference": [Token(None, str(asts[1]), str(asts[1]))],
                    "variables": [["#" + str(asts[1])]],
                    "interpreter_state": BatchedState(
                        {str(asts[1]): None},
                        {str(asts[1]): ["#" + str(asts[1])]},
                        [str(asts[1])],
                        ["#" + str(asts[1])]
                    )
                })),
            1) == samples[1]
        assert DuplicatedSamplerState(
            SamplerState(1.0 / 3, Environment(
                {
                    "test_cases": [(None, None)],
                    "reference": [Token(None, str(asts[2]), str(asts[2]))],
                    "variables": [["#" + str(asts[2])]],
                    "interpreter_state": BatchedState(
                        {str(asts[2]): None},
                        {str(asts[2]): ["#" + str(asts[2])]},
                        [str(asts[2])],
                        ["#" + str(asts[2])]
                    )
                })),
            1) == samples[2]
