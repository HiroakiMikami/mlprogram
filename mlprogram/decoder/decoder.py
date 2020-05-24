from typing import Generic, TypeVar, Generator, Any
from dataclasses import dataclass


Input = TypeVar("Input")
Output = TypeVar("Output")
State = TypeVar("State")


@dataclass
class Result(Generic[Output]):
    output: Output
    score: float


@dataclass
class DecoderState(Generic[State]):
    score: float
    state: State

    def __eq__(self, obj: Any) -> bool:
        if isinstance(obj, DecoderState):
            return self.score == obj.score and self.state == obj.state
        return False

    def __hash__(self) -> int:
        return hash(self.score) ^ hash(self.state)


class Decoder(Generic[Input, Output]):
    def __call__(self, input: Input) -> Generator[Result[Output], None, None]:
        raise NotImplementedError
