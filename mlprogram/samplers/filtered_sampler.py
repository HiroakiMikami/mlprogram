from typing \
    import TypeVar, Generic, Optional, List, Generator, Callable, Tuple
from mlprogram.samplers.sampler \
    import Sampler, SamplerState, DuplicatedSamplerState
from mlprogram.utils import logging


Input = TypeVar("Input")
Output = TypeVar("Output")
Output1 = TypeVar("Output1")
Output2 = TypeVar("Output2")
State = TypeVar("State")

logger = logging.Logger(__name__)


class FilteredSampler(Generic[Input, Output, State]):
    def __init__(self, sampler: Sampler[Input, Output, State],
                 score: Callable[[Input, Output], float],
                 threshold: float,
                 retain_not_finished_output: bool
                 ):
        self.sampler = sampler
        self.score = score
        self.threshold = threshold
        self.retain_not_finished_output = retain_not_finished_output

    def initialize(self, input: Input) -> State:
        return self.sampler.initialize(input)

    def create_output(self, input: Input, state: State) \
            -> Optional[Tuple[Output, bool]]:
        output_opt = self.sampler.create_output(input, state)
        if output_opt is None:
            return None
        output, is_finished = output_opt
        with logger.block("score"):
            score = self.score(input, output)
        if score >= self.threshold:
            logger.debug(f"find appropriate output: score={score}")
            is_finished = True
        if not self.retain_not_finished_output and not is_finished:
            return None
        return output, is_finished

    def top_k_samples(self, states: List[SamplerState[State]], k: int) \
            -> Generator[DuplicatedSamplerState[State], None, None]:
        return self.sampler.top_k_samples(states, k)

    def k_samples(self, states: List[SamplerState[State]], n: List[int]) \
            -> Generator[DuplicatedSamplerState[State], None, None]:
        return self.sampler.k_samples(states, n)
