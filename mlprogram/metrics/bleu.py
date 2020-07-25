from typing import Dict, Any
from nltk.translate.bleu_score import sentence_bleu
from .metric import Metric


class Bleu(Metric[str]):
    def __call__(self, input: Dict[str, Any], value: str) -> float:
        gts = input["ground_truth"]
        return sentence_bleu(list(gts), value)
