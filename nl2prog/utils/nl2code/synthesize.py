import torch
from typing import List, Callable
from nl2prog.utils.data.nl2code import Query
from nl2prog.language.ast import AST

from .beam_search_synthesizer import BeamSearchSynthesizer, Progress


def synthesize(query: Query,
               encode_query: Callable[[List[str]], torch.Tensor],
               synthesizer: BeamSearchSynthesizer
               ) -> [List[Progress], List[AST]]:
    query_embeddings = encode_query(query.query_for_dnn)
    candidates = []
    progress = []
    for cs, ps in synthesizer.synthesize(query.query_for_synth,
                                         query_embeddings):
        candidates.extend(cs)
        progress.append(ps)
    candidates = sorted(candidates, key=lambda x: x.score, reverse=True)
    return progress, [c.ast for c in candidates]
