from typing import List, Tuple
from mlprogram.action.ast import AST

from mlprogram.synthesizer import Synthesizer, Progress


def synthesize(query: str,
               synthesizer: Synthesizer
               ) -> Tuple[List[List[Progress]], List[AST]]:
    candidates = []
    progress = []
    for cs, ps in synthesizer.synthesize(query):
        candidates.extend(cs)
        progress.append(ps)
    candidates = sorted(candidates, key=lambda x: x.score, reverse=True)
    return progress, [c.ast for c in candidates]
