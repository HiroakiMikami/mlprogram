from typing import Callable, Generator, Generic, List, Optional, Tuple, TypeVar

import numpy as np
import torch
from torch import nn

from mlprogram import Environment, logging
from mlprogram.languages import BatchedState, Expander, Interpreter, Token
from mlprogram.samplers import DuplicatedSamplerState, Sampler, SamplerState
from mlprogram.synthesizers.synthesizer import Synthesizer
from mlprogram.utils.data import Collate

logger = logging.Logger(__name__)

Input = TypeVar("Input")
Code = TypeVar("Code")
Value = TypeVar("Value")
Kind = TypeVar("Kind")


class SequentialProgramSampler(Sampler[Input, Code, Environment],
                               Generic[Input, Code, Value, Kind]):
    def __init__(self,
                 synthesizer: Synthesizer[Environment, Code],
                 transform_input: Callable[[Input], Environment],
                 collate: Collate,
                 encoder: nn.Module,
                 expander: Expander[Code],
                 interpreter: Interpreter[Code, Input, Value, Kind],
                 rng: Optional[np.random.RandomState] = None):
        self.synthesizer = synthesizer
        self.transform_input = transform_input
        self.collate = collate
        self.encoder = encoder
        self.expander = expander
        self.interpreter = interpreter
        self.rng = \
            rng or np.random.RandomState(np.random.randint(0, 2 << 32 - 1))

    @logger.function_block("initialize")
    def initialize(self, input: Input) -> Environment:
        self.encoder.eval()
        state_list = self.transform_input(input)
        state_tensor = self.collate.collate([state_list])
        with torch.no_grad(), logger.block("encode_state"):
            state_tensor = self.encoder(state_tensor)
        state = self.collate.split(state_tensor)[0]
        state.states["reference"] = []
        state.states["variables"] = []
        state.states["interpreter_state"] = \
            BatchedState[Code, Value, Kind]({}, {}, [])
        return state

    def create_output(self, input: Input, state: Environment) \
            -> Optional[Tuple[Code, bool]]:
        if len(state.states["interpreter_state"].history) == 0:
            return None
        code = self.expander.unexpand(
            state.states["interpreter_state"].history
        )
        return code, False

    def batch_k_samples(self, states: List[SamplerState[Environment]],
                        ks: List[int]) \
            -> Generator[DuplicatedSamplerState[Environment],
                         None, None]:
        assert all([len(state.state.outputs) for state in states]) == 0
        assert all([len(state.state.supervisions) for state in states]) == 0

        orig_inputs = [state.state.inputs.to_dict() for state in states]
        orig_states = [state.state.states.to_dict() for state in states]

        with logger.block("batch_k_samples"):
            for orig_input, orig_state, state, k in zip(orig_inputs,
                                                        orig_states, states,
                                                        ks):
                if k == 0:
                    continue

                test_cases = state.state.inputs["test_cases"]
                inputs = [input for input, _ in test_cases]
                cnt = 0
                for result in self.synthesizer(state.state,
                                               n_required_output=k):

                    # TODO recompute state (?)
                    new_state = Environment(inputs=orig_input,
                                            states=orig_state)
                    # Clear reference and variables
                    new_state.states["reference"] = []
                    new_state.states["variables"] = []
                    new_state.states["interpreter_state"] = \
                        self.interpreter.execute(
                            result.output, inputs,
                            state.state.states["interpreter_state"]
                    )
                    for code in \
                            new_state.states["interpreter_state"].environment:
                        new_state.states["reference"].append(
                            Token[Kind, Code](
                                new_state.states["interpreter_state"]
                                .type_environment[code],
                                code, code)
                        )
                        new_state.states["variables"].append(
                            new_state.states["interpreter_state"]
                            .environment[code]
                        )
                    yield DuplicatedSamplerState(
                        SamplerState(result.score, new_state), result.num)
                    cnt += 1
                    if cnt == k:
                        break
