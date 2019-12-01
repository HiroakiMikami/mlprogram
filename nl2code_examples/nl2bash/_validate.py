import torch
from dataclasses import dataclass
from typing import List, Union, Callable
from nltk.translate.bleu_score import sentence_bleu

from nl2code import BeamSearchSynthesizer, Progress
from nl2code.language.ast import AST
from nl2code_examples.nl2bash import unparse


def bleu4(references, candidate):
    return sentence_bleu(references, candidate)


@dataclass
class Score:
    is_exact_match: bool
    is_match: bool
    bleu: float
    normalized_bleu: float


@dataclass
class Result:
    query: List[str]
    ground_truth: List[str]
    progress: List[Progress]
    candidates: Union[List[AST], None]
    top1_score: Score
    top3_score: Score


def validate(query: List[str],
             ground_truth: List[str],
             normalized_ground_truth: List[str],
             encoder: Callable[[List[str]], torch.FloatTensor],
             synthesizer: BeamSearchSynthesizer) -> Result:
    """
    Validate the model by using the synthesizer

    Parameters
    ----------
    dataset: EvalDataset
        The dataset for validation. Each item contains
        the tuple of query (List[str]) and ground truth (str).
    encoder: Callable[[List[str]], torch.FloatTensor]
        The function to convert query (with placeholders) to query embeddings.
    synthesizer: BeamSearchSynthesizer

    Returns
    ------
    Result
        The validation result
    """
    candidates = []
    progress = []
    for cs, ps in synthesizer.synthesize(
            query, encoder(query)):
        candidates.extend(cs)
        progress.append(ps)
    candidates.sort(key=lambda x: -x.score)

    # Calc top-1 score
    if len(candidates) == 0:
        top1 = Score(False, False, 0, 0)
    else:
        c = candidates[0]
        code = unparse(c.ast)
        top1 = Score(code in ground_truth, code in normalized_ground_truth,
                     bleu4(ground_truth, code),
                     bleu4(normalized_ground_truth, code)
                     if len(normalized_ground_truth) != 0 else None)

    # Calc top-3 score
    top3 = Score(False, False, 0, None)
    for c in candidates[:3]:
        code = unparse(c.ast)
        s = Score(code in ground_truth, code in normalized_ground_truth,
                  bleu4(ground_truth, code),
                  bleu4(normalized_ground_truth, code)
                  if len(normalized_ground_truth) != 0 else None)
        top3.is_exact_match = top3.is_exact_match or s.is_exact_match
        top3.is_match = top3.is_match or s.is_match
        top3.bleu = max(top3.bleu, s.bleu)
        if s.normalized_bleu is not None:
            if top3.normalized_bleu is None:
                top3.normalized_bleu = s.normalized_bleu
            else:
                top3.normalized_bleu = max(top3.normalized_bleu,
                                           s.normalized_bleu)

    return Result(query, ground_truth, progress, [c.ast for c in candidates],
                  top1, top3)
