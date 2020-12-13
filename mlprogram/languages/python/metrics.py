import re
from typing import List

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from torch import nn


class Bleu(nn.Module):
    def forward(self, expected: str, actual: str) -> float:
        sm = SmoothingFunction()

        def tokenize(code: str) -> List[str]:
            code = re.sub(r'([^A-Za-z0-9_])', r' \1 ', code)
            code = re.sub(r'([a-z])([A-Z])', r'\1 \2', code)
            code = re.sub(r'\s+', ' ', code)
            code = re.sub(r'["\']', '`', code)
            tokens = [t for t in code.split(' ') if t]
            return tokens

        ref = [tokenize(expected)]
        cand = tokenize(actual)
        return float(sentence_bleu(ref,
                                   cand,
                                   weights=[0.25] * min(4, len(ref)),
                                   smoothing_function=sm.method3))
