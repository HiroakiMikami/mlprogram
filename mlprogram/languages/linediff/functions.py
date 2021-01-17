from typing import List, Union, cast

from torch import nn

from mlprogram.builtins import Environment
from mlprogram.languages import Kinds, Root
from mlprogram.languages.linediff.expander import Expander
from mlprogram.languages.linediff.interpreter import Interpreter


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


class ToEpisode(nn.Module):
    def __init__(self, interpreter: Interpreter, expander: Expander):
        super().__init__()
        self.interpreter = interpreter
        self.expander = expander

    def forward(self, entry: Environment) -> List[Environment]:
        ground_truth = entry["ground_truth"]
        inputs = [input for input, _ in entry["test_cases"]]

        retval: List[Environment] = []
        state = self.interpreter.create_state(inputs)
        for code in self.expander.expand(ground_truth):
            xs = cast(Environment, entry.clone())
            xs["ground_truth"] = code
            xs["reference"] = []
            xs["variables"] = []
            xs["interpreter_state"] = state
            state = self.interpreter.execute(code, state)
            retval.append(xs)
        return retval


class AddTestCases(nn.Module):
    def forward(self, entry: Environment) -> Environment:
        entry = cast(Environment, entry.clone())
        if "test_cases" in entry:
            return entry
        query = entry["code"]
        entry["test_cases"] = [(query, None)]
        return entry


class UpdateInput(nn.Module):
    def forward(self, entry: Environment) -> Environment:
        entry = cast(Environment, entry.clone())
        state = entry["interpreter_state"]
        inputs = state.context
        code = inputs[0]
        entry["code"] = code

        return entry
