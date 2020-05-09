from typing import Union, Callable, Set, List, Optional
from mlprogram.ast.ast import AST
import logging

logger = logging.getLogger()

Element = Union[AST, str]


class Metric:
    def __init__(self, parse: Callable[[str], AST],
                 unparse: Callable[[AST], str],
                 metric: Callable[[Set[str], str], float]):
        self.parse = parse
        self.unparse = unparse
        self.metric = metric

    def __call__(self, ground_truths: List[Element], value: Element) -> float:
        gts = set()
        for gt in ground_truths:
            if not isinstance(gt, str):
                code: Optional[str] = self.unparse(gt)
                if code is not None:
                    gts.add(self.unparse(gt))
            else:
                gts.add(gt)
                node = self.parse(gt)
                if node is not None:
                    code = self.unparse(node)
                else:
                    code = None
                if code is not None:
                    gts.add(code)
        if not isinstance(value, str):
            code = self.unparse(value)
            if code is None:
                value = str(value)
            else:
                value = code
        return self.metric(gts, value)
