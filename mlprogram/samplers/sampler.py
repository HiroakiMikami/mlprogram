from typing import TypeVar, Generic, Optional, List, Generator, Any
from dataclasses import dataclass


Input = TypeVar("Input")
Output = TypeVar("Output")
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


class Sampler(Generic[Input, Output, State]):
    def initialize(self, input: Input) -> State:
        raise NotImplementedError

    def create_output(self, state: State) -> Optional[Output]:
        raise NotImplementedError

    def top_k_samples(self, states: List[SamplerState[State]], k: int) \
            -> Generator[SamplerState[State], None, None]:
        raise NotImplementedError

    def random_samples(self, state: SamplerState[State], n: int) \
            -> Generator[SamplerState[State], None, None]:
        raise NotImplementedError