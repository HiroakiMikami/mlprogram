from typing import Generator, Generic, Optional, TypeVar

from mlprogram import logging
from mlprogram.samplers import Sampler, SamplerState
from mlprogram.synthesizers.synthesizer import Result, Synthesizer

logger = logging.Logger(__name__)

Input = TypeVar("Input")
Output = TypeVar("Output")
State = TypeVar("State")


class BeamSearch(Synthesizer[Input, Output], Generic[Input, Output, State]):
    def __init__(self,
                 beam_size: int, max_step_size: int,
                 sampler: Sampler[Input, Output, State]):
        self.beam_size = beam_size
        self.max_step_size = max_step_size
        self.sampler = sampler

    @logger.function_block("__call__")
    def __call__(self, input: Input, n_required_output: Optional[int] = None) \
            -> Generator[Result[Output], None, None]:
        # Start from empty sequence
        states = [SamplerState(0.0, self.sampler.initialize(input))]

        k = self.beam_size
        steps = 0
        with logger.block("__call__"):
            while steps < self.max_step_size and k > 0:
                if len(states) == 0:
                    return
                next_states = []

                for next_state in self.sampler.top_k_samples(states, k):
                    output_opt = self.sampler.create_output(
                        input, next_state.state.state)
                    if output_opt is not None:
                        output, is_finished = output_opt
                        if steps == self.max_step_size - 1:
                            # The step is last
                            is_finished = True
                        yield Result(output, next_state.state.score,
                                     is_finished, 1)
                        if is_finished:
                            k -= 1
                        else:
                            next_states.append(next_state.state)
                    else:
                        next_states.append(next_state.state)
                states = next_states
                steps += 1
