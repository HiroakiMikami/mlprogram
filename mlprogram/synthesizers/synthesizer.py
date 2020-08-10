from typing import Generic, TypeVar, Generator, Optional
from dataclasses import dataclass


Input = TypeVar("Input")
Output = TypeVar("Output")
State = TypeVar("State")


@dataclass
class Result(Generic[Output]):
    output: Output
    score: float


class Synthesizer(Generic[Input, Output]):
    def __call__(self, input: Input, n_required_output: Optional[int] = None) \
            -> Generator[Result[Output], None, None]:
        raise NotImplementedError
