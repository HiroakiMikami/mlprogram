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
        inputs = [input for input, _ in entry.inputs["test_cases"]]

        retval: List[Environment] = []
        state = BatchedState[linediffAST, str, str]({}, {}, [])
        for code in self.expander.expand(ground_truth):
            xs = entry.clone()
            xs.supervisions["ground_truth"] = code
            state = self.interpreter.execute(code, inputs, state)
            next_inputs = list(state.environment.values())[0]
            xs.inputs["test_cases"] = list(zip(inputs, next_inputs))
            inputs = next_inputs
            retval.append(xs)
        return retval


# TODO remove this class
class AddTestCases:
    def __call__(self, entry: Environment) -> Environment:
        if "test_cases" in entry.inputs:
            return entry
        query = entry.inputs["code"]
        entry.inputs.mutable(True)  # TODO
        entry.inputs["test_cases"] = [(query, None)]
        return entry


# TODO remove this class
class UpdateInput:
    def __call__(self, entry: Environment) -> Environment:
        entry.inputs.mutable(True)  # TODO
        if "interpreter_state" in entry.states \
                and len(entry.states["interpreter_state"].history) > 0:
            state = entry.states["interpreter_state"]
            inputs = state.environment[state.history[-1]]
        else:
            inputs = [input for input, _ in entry.inputs["test_cases"]]
        code = inputs[0]
        entry.inputs["code"] = code
        entry.inputs["text_query"] = code

        return entry
