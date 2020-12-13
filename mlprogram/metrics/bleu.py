from nltk.translate.bleu_score import sentence_bleu
from torch import nn


class Bleu(nn.Module):
    def forward(self, expected: str, actual: str) -> float:
        return float(sentence_bleu([expected], actual))
