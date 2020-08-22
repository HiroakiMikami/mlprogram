from typing \
    import TypeVar, Generic, Optional, List, Generator, Any, Callable, Tuple
from dataclasses import dataclass


Input = TypeVar("Input")
Output = TypeVar("Output")
Output1 = TypeVar("Output1")
Output2 = TypeVar("Output2")
State = TypeVar("State")


@dataclass
class SamplerState(Generic[State]):
    score: float
    state: State

    def __eq__(self, obj: Any) -> bool:
        if isinstance(obj, SamplerState):
            return self.score == obj.score and self.state == obj.state
        return False

    def __hash__(self) -> int:
        return hash(self.score) ^ hash(self.state)


@dataclass
class DuplicatedSamplerState(Generic[State]):
    state: SamplerState[State]
    num: int

    def __eq__(self, obj: Any) -> bool:
        if isinstance(obj, DuplicatedSamplerState):
            return self.state == obj.state and self.num == obj.num
        return False

    def __hash__(self) -> int:
        return hash(self.state) ^ hash(self.num)


class Sampler(Generic[Input, Output, State]):
    def initialize(self, input: Input) -> State:
        raise NotImplementedError

    def create_output(self, input: Input, state: State) \
            -> Optional[Tuple[Output, bool]]:
        raise NotImplementedError

    def top_k_samples(self, states: List[SamplerState[State]], k: int) \
            -> Generator[DuplicatedSamplerState[State], None, None]:
        raise NotImplementedError

    def k_samples(self, states: List[SamplerState[State]], n: List[int]) \
            -> Generator[DuplicatedSamplerState[State], None, None]:
        raise NotImplementedError


def transform(sampler: Sampler[Input, Output1, State],
              transform: Callable[[Output1], Optional[Output2]]) \
        -> Sampler[Input, Output2, State]:
    class TransformedSampler(Sampler[Input, Output2, State]):
        def __init__(self, sampler: Sampler[Input, Output1, State],
                     transform: Callable[[Output1], Optional[Output2]]):
            self.sampler = sampler
            self.transform = transform

        def initialize(self, input: Input) -> State:
            return self.sampler.initialize(input)

        def create_output(self, input: Input, state: State) \
                -> Optional[Tuple[Output2, bool]]:
            output = self.sampler.create_output(input, state)
            if output is None:
                return None
            else:
                output1, is_finished = output
                output2 = self.transform(output1)
                if output2 is None:
                    return None
                else:
                    return output2, is_finished

        def top_k_samples(self, states: List[SamplerState[State]], k: int) \
                -> Generator[DuplicatedSamplerState[State], None, None]:
            return self.sampler.top_k_samples(states, k)

        def k_samples(self, states: List[SamplerState[State]], n: List[int]) \
                -> Generator[DuplicatedSamplerState[State], None, None]:
            return self.sampler.k_samples(states, n)

    return TransformedSampler(sampler, transform)
