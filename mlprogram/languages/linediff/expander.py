from typing import List

from mlprogram.languages import Expander as BaseExpander
from mlprogram.languages.linediff import AST, Diff


class Expander(BaseExpander[AST]):
    def expand(self, code: AST) -> List[AST]:
        if isinstance(code, Diff):
            return [delta for delta in code.deltas]
        else:
            return [code]

    def unexpand(self, code: List[AST]) -> AST:
        deltas = []
        for elem in code:
            if isinstance(elem, Diff):
                deltas.extend(elem.deltas)
            else:
                deltas.append(elem)
        return Diff(deltas)
