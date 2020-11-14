from typing import Union

from mlprogram import Environment
from mlprogram.encoders import Samples
from mlprogram.utils.data import ListDataset
from mlprogram.utils import data
from mlprogram.languages import Root, Parser
from mlprogram.languages.linediff.ast \
    import AST as linediffAST, Diff, Insert, Replace, Remove


class IsSubtype:
    def __call__(self, subtype: Union[str, Root],
                 basetype: Union[str, Root]) -> bool:
        if isinstance(basetype, Root):
            return True
        if basetype == "Delta":
            return subtype in set(["Delta", "Insert", "Remove", "Replace"])
        return subtype == basetype


def get_samples(parser: Parser[linediffAST]) -> Samples:
    dataset = ListDataset([Environment(
        supervisions={"ground_truth":
                      Diff([Insert(0, "x"), Remove(1), Replace(2, "y")])})
    ])
    samples = data.get_samples(dataset, parser)
    samples.tokens.clear()

    return samples
