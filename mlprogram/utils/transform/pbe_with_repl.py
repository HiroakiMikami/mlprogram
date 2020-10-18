from typing import List
from typing import TypeVar
from typing import Generic
from mlprogram import Environment
from mlprogram.languages import Token
from mlprogram.languages import Expander
from mlprogram.interpreters import State
from mlprogram.interpreters import Interpreter

Code = TypeVar("Code")
Input = TypeVar("Input")
Value = TypeVar("Value")
Kind = TypeVar("Kind")


class ToEpisode(Generic[Code, Input, Value]):
    def __init__(self, interpreter: Interpreter[Code, Input, Value, Kind],
                 expander: Expander[Code]):
        self.interpreter = interpreter
        self.expander = expander

    def __call__(self, entry: Environment) -> List[Environment]:
        ground_truth = entry.supervisions["ground_truth"]
        input, _ = entry.inputs["test_case"]

        retval: List[Environment] = []
        state = State[Code, Value, Kind]({}, {}, [])
        for code in self.expander.expand(ground_truth):
            xs = entry.clone()
            # TODO set type of reference
            xs.states["reference"] = [
                Token(state.type_environment[v], v, v)
                for v in state.environment.keys()
            ]
            xs.states["variables"] = [
                state.environment[token.value]
                for token in xs.states["reference"]
            ]
            xs.supervisions["ground_truth"] = code
            state = self.interpreter.execute(code, input, state)
            retval.append(xs)

        return retval
