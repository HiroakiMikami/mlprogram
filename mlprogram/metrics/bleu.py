from nltk.translate.bleu_score import sentence_bleu

from mlprogram import Environment
from mlprogram.metrics.metric import Metric


class Bleu(Metric[str]):
    def __call__(self, input: Environment, value: str) -> float:
        ground_truth = input["ground_truth"]
        return sentence_bleu([ground_truth], value)
