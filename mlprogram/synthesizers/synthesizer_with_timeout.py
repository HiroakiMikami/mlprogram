import time
from typing import TypeVar, Generic, Generator, Optional
from mlprogram.synthesizers import Synthesizer, Result
from mlprogram import logging

logger = logging.Logger(__name__)

Input = TypeVar("Input")
Output = TypeVar("Output")


class SynthesizerWithTimeout(Synthesizer[Input, Output],
                             Generic[Input, Output]):
    def __init__(self, synthesizer: Synthesizer[Input, Output],
                 timeout_sec: float):
        self.synthesizer = synthesizer
        self.timeout_sec = timeout_sec

    def __call__(self, input: Input, n_required_output: Optional[int] = None) \
            -> Generator[Result[Output], None, None]:
        begin = time.time()
        with logger.block("__call__"):
            for output in self.synthesizer(
                    input,
                    n_required_output=n_required_output):
                yield output
                if time.time() - begin > self.timeout_sec:
                    logger.debug("timeout")
                    break
