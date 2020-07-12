from typing import Callable, List, Generic, TypeVar, Optional
from mlprogram.metrics.metric_using_ground_truth import MetricUsingGroundTruth
import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

Code = TypeVar("Code")
Value = TypeVar("Value")


class Bleu(MetricUsingGroundTruth[Code, Value], Generic[Code, Value]):
    def __init__(self, parse: Optional[Callable[[Code], Value]],
                 unparse: Optional[Callable[[Value], Code]]):
        super().__init__(parse, unparse)

    def metric(self, gts, value) -> float:
        sm = SmoothingFunction()

        def tokenize(code: str) -> List[str]:
            code = re.sub(r'([^A-Za-z0-9_])', r' \1 ', code)
            code = re.sub(r'([a-z])([A-Z])', r'\1 \2', code)
            code = re.sub(r'\s+', ' ', code)
            code = re.sub(r'["\']', '`', code)
            tokens = [t for t in code.split(' ') if t]
            return tokens

        ref = [tokenize(ref) for ref in gts]
        cand = tokenize(value)
        return sentence_bleu(ref,
                             cand,
                             weights=[0.25] * min(4, len(ref)),
                             smoothing_function=sm.method3)
