from typing import List, Tuple
from nl2prog.utils import Progress
from nl2prog.language.ast import AST

from .beam_search_synthesizer import BeamSearchSynthesizer


def synthesize(query: str,
               synthesizer: BeamSearchSynthesizer
               ) -> Tuple[List[Progress], List[AST]]:
    candidates = []
    progress = []
    for cs, ps in synthesizer.synthesize(query):
        candidates.extend(cs)
        progress.append(ps)
    candidates = sorted(candidates, key=lambda x: x.score, reverse=True)
    return progress, [c.ast for c in candidates]
