from typing import Callable, TypeVar, Generic, Optional, Generator, List
from mlprogram.decoder import Decoder, Result, DecoderState

Input = TypeVar("Input")
Output = TypeVar("Output")
State = TypeVar("State")


class BeamSearch(Decoder[Input, Output], Generic[Input, Output, State]):
    def __init__(self,
                 beam_size: int, max_step_size: int,
                 initialize: Callable[[Input], State],
                 create_output: Callable[[State], Optional[Output]],
                 top_k_samples: Callable[[List[DecoderState[State]], int],
                                         Generator[DecoderState[State], None,
                                                   None]]):
        self.beam_size = beam_size
        self.max_step_size = max_step_size
        self.initialize = initialize
        self.create_output = create_output
        self.top_k_samples = top_k_samples

    def __call__(self, input: Input) -> Generator[Result[Output], None, None]:
        # Start from empty sequence
        states = [DecoderState(0.0, self.initialize(input))]

        k = self.beam_size
        steps = 0
        while steps < self.max_step_size and k > 0:
            next_states = []
            for next_state in self.top_k_samples(states, k):
                output_opt = self.create_output(next_state.state)
                if output_opt is not None:
                    yield Result(output_opt, next_state.score)
                    k -= 1
                else:
                    next_states.append(next_state)
            states = next_states
            steps += 1
