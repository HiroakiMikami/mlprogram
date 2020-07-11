from typing import Callable, Generic, TypeVar
from nltk.translate.bleu_score import sentence_bleu
from .metric_using_ground_truth import MetricUsingGroundTruth

Code = TypeVar("Code")
Value = TypeVar("Value")


class Bleu(MetricUsingGroundTruth[Code, Value], Generic[Code, Value]):
    def __init__(self, parse: Callable[[Code], Value],
                 unparse: Callable[[Value], Code]):
        super().__init__(parse, unparse)

    def metric(self, gts, value) -> float:
        return sentence_bleu(list(gts), value)
