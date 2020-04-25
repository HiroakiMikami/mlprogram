from typing import Callable, List, Iterable
import re
from nl2prog.language.ast import AST
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from nl2prog.metrics import Metric


def bleu(reference: Iterable[str], candidate: str) -> float:
    sm = SmoothingFunction()

    def tokenize(code: str) -> List[str]:
        code = re.sub(r'([^A-Za-z0-9_])', r' \1 ', code)
        code = re.sub(r'([a-z])([A-Z])', r'\1 \2', code)
        code = re.sub(r'\s+', ' ', code)
        code = re.sub(r'["\']', '`', code)
        tokens = [t for t in code.split(' ') if t]
        return tokens

    ref = [tokenize(ref) for ref in reference]
    cand = tokenize(candidate)
    return sentence_bleu(ref,
                         cand,
                         weights=[0.25] * min(4, len(ref)),
                         smoothing_function=sm.method3)


class Bleu(Metric):
    def __init__(self, parse: Callable[[str], AST],
                 unparse: Callable[[AST], str]):
        super(Bleu, self).__init__(parse, unparse, bleu)
