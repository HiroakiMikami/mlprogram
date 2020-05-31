from typing import Callable
from mlprogram.ast import AST
from nltk.translate.bleu_score import sentence_bleu

from .metric import Metric


class Bleu(Metric):
    def __init__(self, parse: Callable[[str], AST],
                 unparse: Callable[[AST], str]):
        super(Bleu, self).__init__(
            parse, unparse,
            lambda gts, value: sentence_bleu(list(gts), value)
        )
