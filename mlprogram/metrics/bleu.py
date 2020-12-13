from nltk.translate.bleu_score import sentence_bleu

from mlprogram.builtins import Environment
from mlprogram.metrics.metric import Metric


class Bleu(Metric[str]):
    def __call__(self, input: Environment, value: str) -> float:
        ground_truth = input["ground_truth"]
        return float(sentence_bleu([ground_truth], value))
