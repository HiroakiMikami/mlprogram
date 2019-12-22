import torch
from dataclasses import dataclass
from typing import List, Union, Callable
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import re

from nl2prog.utils.nl2code import BeamSearchSynthesizer, Progress
from nl2prog.language.ast import AST
from nl2prog.language.python import to_python_ast
from nl2code_examples.hearthstone import unparse


def bleu4(reference, candidate):
    sm = SmoothingFunction()

    def tokenize(code):
        code = re.sub(r'([^A-Za-z0-9_])', r' \1 ', code)
        code = re.sub(r'([a-z])([A-Z])', r'\1 \2', code)
        code = re.sub(r'\s+', ' ', code)
        code = re.sub(r'["\']', '`', code)
        tokens = [t for t in code.split(' ') if t]
        return tokens

    ref = tokenize(reference)
    cand = tokenize(candidate)
    return sentence_bleu([ref],
                         cand,
                         weights=[0.25] * min(4, len(ref)),
                         smoothing_function=sm.method3)


@dataclass
class Result:
    query: List[str]
    ground_truth: str
    progress: List[Progress]
    candidates: Union[AST, None]
    is_match: bool
    bleu4: float


def validate(query: List[str],
             ground_truth: str,
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
    candidate = None
    for c in candidates:
        if candidate is None:
            candidate = c
        else:
            if candidate.score < c.score:
                candidate = c
    if candidate is None:
        code = ""
        return Result(query, ground_truth, progress, None,
                      ground_truth == code, bleu4(ground_truth, code))
    else:
        try:
            code = unparse(to_python_ast(candidate.ast))
        except:  # noqa
            code = ""

        return Result(query, ground_truth, progress, candidate.ast,
                      ground_truth == code, bleu4(ground_truth, code))
