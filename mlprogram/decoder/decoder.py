from typing import Generic, TypeVar, Generator
from dataclasses import dataclass


Input = TypeVar("Input")
Output = TypeVar("Output")
State = TypeVar("State")


@dataclass
class Result(Generic[Output]):
    output: Output
    score: float


class Decoder(Generic[Input, Output]):
    def __call__(self, input: Input) -> Generator[Result[Output], None, None]:
        raise NotImplementedError
