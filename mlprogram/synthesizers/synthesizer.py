from dataclasses import dataclass
from typing import Generator, Generic, Optional, TypeVar

Input = TypeVar("Input")
Output = TypeVar("Output")
State = TypeVar("State")


@dataclass
class Result(Generic[Output]):
    output: Output
    score: float
    is_finished: bool
    num: int


class Synthesizer(Generic[Input, Output]):
    def __call__(self, input: Input, n_required_output: Optional[int] = None) \
            -> Generator[Result[Output], None, None]:
        for idx, output in enumerate(self._synthesize(input, n_required_output)):
            yield output
            if (idx + 1) == n_required_output:
                break

    def _synthesize(self, input: Input, n_required_output: Optional[int]) \
            -> Generator[Result[Output], None, None]:
        raise NotImplementedError
