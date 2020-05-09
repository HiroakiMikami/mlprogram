from typing import Callable
from mlprogram.ast.ast import AST

from .metric import Metric


class Accuracy(Metric):
    def __init__(self, parse: Callable[[str], AST],
                 unparse: Callable[[AST], str]):
        super(Accuracy, self).__init__(
            parse, unparse,
            lambda gts, value: 1.0 if value in gts else 0.0
        )
