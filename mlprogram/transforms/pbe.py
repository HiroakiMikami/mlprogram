from typing import Generic, List, TypeVar

from torch import nn

from mlprogram.builtins import Environment
from mlprogram.languages import Expander, Interpreter, Token

Code = TypeVar("Code")
Input = TypeVar("Input")
Value = TypeVar("Value")
Kind = TypeVar("Kind")
Context = TypeVar("Context")


class ToEpisode(nn.Module, Generic[Code, Input, Value, Context]):
    def __init__(self, interpreter: Interpreter[Code, Input, Value, Kind, Context],
                 expander: Expander[Code]):
        super().__init__()
        self.interpreter = interpreter
        self.expander = expander

    def forward(self, entry: Environment) -> List[Environment]:
        ground_truth = entry["ground_truth"]
        test_cases = entry["test_cases"]
        inputs = [input for input, _ in test_cases]

        retval: List[Environment] = []
        state = self.interpreter.create_state(inputs)
        for code in self.expander.expand(ground_truth):
            xs = entry.clone()
            xs["reference"] = [
                Token(state.type_environment[v], v, v)
                for v in state.environment.keys()
            ]
            xs["variables"] = [
                state.environment[token.value]
                for token in xs["reference"]
            ]
            xs["ground_truth"] = code
            xs.mark_as_supervision("ground_truth")
            state = self.interpreter.execute(code, state)
            retval.append(xs)

        return retval
