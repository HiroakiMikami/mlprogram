from typing import TypeVar, Generic, Generator, Callable, Optional
from mlprogram.synthesizers import Synthesizer, Result
from mlprogram.utils import TopKElement

Input = TypeVar("Input")
Output = TypeVar("Output")


class FilteredSynthesizer(Synthesizer[Input, Output], Generic[Input, Output]):
    def __init__(self, synthesizer: Synthesizer[Input, Output],
                 score: Callable[[Input, Output], float],
                 threshold: float,
                 n_output_if_empty: int = 0,
                 metric: str = "score"):
        self.synthesizer = synthesizer
        self.score = score
        self.threshold = threshold
        self.n_output_if_empty = n_output_if_empty
        assert metric in set(["score", "original_score"])
        self.metric = metric

    def __call__(self, input: Input, n_required_output: Optional[int] = None) \
            -> Generator[Result[Output], None, None]:
        topk = TopKElement(self.n_output_if_empty)
        for result in self.synthesizer(input, n_required_output):
            original_score = result.score
            score = self.score(input, result.output)
            if score >= self.threshold:
                yield result
                return
            s = score if self.metric == "score" else original_score
            topk.add(s, result.output)
        for elem in topk.elements:
            yield Result(elem[1], elem[0])
