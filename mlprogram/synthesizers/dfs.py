from typing \
    import TypeVar, Generic, Optional, Generator
from mlprogram.samplers import Sampler, SamplerState, DuplicatedSamplerState
from mlprogram.synthesizers import Result, Synthesizer
from mlprogram import logging

logger = logging.Logger(__name__)

Input = TypeVar("Input")
Output = TypeVar("Output")
State = TypeVar("State")
Key = TypeVar("Key")


class DFS(Synthesizer[Input, Output], Generic[Input, Output, State, Key]):
    def __init__(self, sampler: Sampler[Input, Output, State]):
        self.sampler = sampler

    def _search(self, input: Input, state: DuplicatedSamplerState[State]) \
            -> Generator[Result[Output], None, None]:
        if state.num == 0:
            return

        for next_state in self.sampler.all_samples([state.state], sorted=True):
            output_opt = \
                self.sampler.create_output(input, next_state.state.state)
            if output_opt is not None:
                output, is_finished = output_opt
                yield Result(output, next_state.state.score,
                             is_finished, 1)
            else:
                is_finished = False
            if not is_finished:
                for result in self._search(input, next_state):
                    yield result

    def __call__(self, input: Input, n_required_output: Optional[int] = None) \
            -> Generator[Result[Output], None, None]:
        with logger.block("__call__"):
            initial_state = DuplicatedSamplerState(
                SamplerState(0.0, self.sampler.initialize(input)), 1
            )
            for output in self._search(input, initial_state):
                yield output
