from typing import List, Union

from mlprogram import Environment
from mlprogram.encoders import Samples
from mlprogram.languages import BatchedState, Kinds, Parser, Root
from mlprogram.languages.linediff.ast import AST as linediffAST
from mlprogram.languages.linediff.ast import Diff, Insert, Remove, Replace
from mlprogram.languages.linediff.expander import Expander
from mlprogram.languages.linediff.interpreter import Interpreter
from mlprogram.utils import data
from mlprogram.utils.data import ListDataset


class IsSubtype:
    def __call__(self, subtype: Union[str, Root],
                 basetype: Union[str, Root]) -> bool:
        if isinstance(basetype, Root):
            return True
        if basetype == "Delta":
            return subtype in set(["Delta", "Insert", "Remove", "Replace"])
        if basetype == "value" and not isinstance(subtype, Kinds.LineNumber):
            return True
        return subtype == basetype


def get_samples(parser: Parser[linediffAST]) -> Samples:
    dataset = ListDataset([Environment(
        supervisions={"ground_truth":
                      Diff([Insert(0, "x"), Remove(1), Replace(2, "y")])})
    ])
    samples = data.get_samples(dataset, parser)
    samples.tokens.clear()

    return samples


class ToEpisode:
    def __init__(self, interpreter: Interpreter, expander: Expander):
        self.interpreter = interpreter
        self.expander = expander

    def __call__(self, entry: Environment) -> List[Environment]:
        ground_truth = entry.supervisions["ground_truth"]
        inputs = [entry.inputs["text_query"]]

        retval: List[Environment] = []
        state = BatchedState[linediffAST, str, str]({}, {}, [])
        for code in self.expander.expand(ground_truth):
            xs = entry.clone()
            xs.inputs["code"] = inputs[0]
            xs.inputs["text_query"] = inputs[0]
            xs.supervisions["ground_truth"] = code
            state = self.interpreter.execute(code, inputs, state)
            inputs = list(state.environment.values())[0]
            retval.append(xs)
        return retval
