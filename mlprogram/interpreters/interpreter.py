from typing import Dict
from typing import Any
from typing import Generic
from typing import List
from typing import TypeVar
from typing import Optional
from dataclasses import dataclass


Code = TypeVar("Code")
Input = TypeVar("Input")
Value = TypeVar("Value")
Kind = TypeVar("Kind")


@dataclass
class State(Generic[Code, Value, Kind]):
    type_environment: Dict[Code, Optional[Kind]]
    environment: Dict[Code, Value]
    history: List[Code]

    def clone(self):
        return State(dict(self.type_environment), dict(self.environment),
                     list(self.history))

    def __hash__(self) -> int:
        return hash(tuple(self.history))

    def __eq__(self, rhs: Any) -> bool:
        if not isinstance(rhs, State):
            return False
        return self.environment == rhs.environment and \
            self.type_environment == rhs.type_environment and \
            self.history == rhs.history


class Interpreter(Generic[Code, Input, Value, Kind]):
    def eval(self, code: Code, input: Input) -> Value:
        raise NotImplementedError

    def execute(self, code: Code, input: Input,
                state: State[Code, Value, Kind]) \
            -> State[Code, Value, Kind]:
        value = self.eval(code, input)
        next = state.clone()
        next.history.append(code)
        next.type_environment[code] = None
        next.environment[code] = value
        return next
