from typing import TypeVar, Generic, Generator, Callable, Optional
from mlprogram.synthesizers import Synthesizer, Result
from mlprogram.utils import logging

logger = logging.Logger(__name__)

Input = TypeVar("Input")
Output = TypeVar("Output")


class FilteredSynthesizer(Synthesizer[Input, Output], Generic[Input, Output]):
    def __init__(self, synthesizer: Synthesizer[Input, Output],
                 score: Callable[[Input, Output], float],
                 threshold: float):
        self.synthesizer = synthesizer
        self.score = score
        self.threshold = threshold

    def __call__(self, input: Input, n_required_output: Optional[int] = None) \
            -> Generator[Result[Output], None, None]:
        with logger.block("__call__"):
            for result in self.synthesizer(input, n_required_output):
                score = self.score(input, result.output)
                if score >= self.threshold:
                    logger.debug(f"find appropriate output: score={score}")
                    yield result
                    return
