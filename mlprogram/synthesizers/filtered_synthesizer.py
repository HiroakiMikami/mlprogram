from typing import Callable, Generator, Generic, Optional, TypeVar

from mlprogram import logging
from mlprogram.synthesizers.synthesizer import Result, Synthesizer

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

    def _synthesize(self, input: Input, n_required_output: Optional[int] = None) \
            -> Generator[Result[Output], None, None]:
        with logger.block("_synthesize"):
            for result in self.synthesizer(input, n_required_output):
                score = self.score(input, result.output)
                if score >= self.threshold:
                    logger.debug(f"find appropriate output: score={score}")
                    yield result
                    return
