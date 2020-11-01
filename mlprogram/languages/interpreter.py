from dataclasses import dataclass
from typing import Any, Dict, Generic, List, Optional, TypeVar

Code = TypeVar("Code")
Input = TypeVar("Input")
Value = TypeVar("Value")
Kind = TypeVar("Kind")


@dataclass
class BatchedState(Generic[Code, Value, Kind]):
    type_environment: Dict[Code, Optional[Kind]]
    environment: Dict[Code, List[Value]]
    history: List[Code]

    def clone(self):
        return BatchedState(dict(self.type_environment),
                            dict(self.environment),
                            list(self.history))

    def __hash__(self) -> int:
        return hash(tuple(self.history))

    def __eq__(self, rhs: Any) -> bool:
        if not isinstance(rhs, BatchedState):
            return False
        return self.environment == rhs.environment and \
            self.type_environment == rhs.type_environment and \
            self.history == rhs.history


class Interpreter(Generic[Code, Input, Value, Kind]):
    def eval(self, code: Code, input: List[Input]) -> List[Value]:
        raise NotImplementedError

    def execute(self, code: Code, input: List[Input],
                state: BatchedState[Code, Value, Kind]) \
            -> BatchedState[Code, Value, Kind]:
        value = self.eval(code, input)
        next = state.clone()
        next.history.append(code)
        next.type_environment[code] = None
        next.environment[code] = value
        return next
