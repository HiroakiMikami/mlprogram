from typing import Dict, Any
from nltk.translate.bleu_score import sentence_bleu
from mlprogram.metrics.metric import Metric


class Bleu(Metric[str]):
    def __call__(self, input: Dict[str, Any], value: str) -> float:
        ground_truth = input["ground_truth"]
        return sentence_bleu([ground_truth], value)
