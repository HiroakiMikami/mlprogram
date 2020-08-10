from typing import TypeVar, Generic, Generator, Optional
from mlprogram.samplers import Sampler, SamplerState
from mlprogram.synthesizers import Synthesizer, Result

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

    def __call__(self, input: Input, n_required_output: Optional[int] = None) \
            -> Generator[Result[Output], None, None]:
        # Start from empty sequence
        states = [SamplerState(0.0, self.sampler.initialize(input))]

        k = self.beam_size
        steps = 0
        while steps < self.max_step_size and k > 0:
            if len(states) == 0:
                return
            next_states = []

            for next_state in self.sampler.top_k_samples(states, k):
                output_opt = self.sampler.create_output(next_state.state)
                if output_opt is not None:
                    yield Result(output_opt, next_state.score)
                    k -= 1
                else:
                    next_states.append(next_state)
            states = next_states
            steps += 1
