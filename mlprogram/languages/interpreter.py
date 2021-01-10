from dataclasses import dataclass
from typing import Any, Dict, Generic, List, Optional, TypeVar

Code = TypeVar("Code")
Input = TypeVar("Input")
Value = TypeVar("Value")
Kind = TypeVar("Kind")
Context = TypeVar("Context")


@dataclass
class BatchedState(Generic[Code, Value, Kind, Context]):
    type_environment: Dict[Code, Optional[Kind]]
    environment: Dict[Code, List[Value]]
    history: List[Code]
    context: List[Context]

    def clone(self):
        return BatchedState(dict(self.type_environment),
                            dict(self.environment),
                            list(self.history),
                            list(self.context))

    def __hash__(self) -> int:
        return hash(tuple(self.history))

    def __eq__(self, rhs: Any) -> bool:
        if not isinstance(rhs, BatchedState):
            return False
        return self.environment == rhs.environment and \
            self.type_environment == rhs.type_environment and \
            self.history == rhs.history and \
            self.context == rhs.context


class Interpreter(Generic[Code, Input, Value, Kind, Context]):
    def eval(self, code: Code, input: List[Input]) -> List[Value]:
        raise NotImplementedError

    def create_state(self, input: List[Input]) \
            -> BatchedState[Code, Value, Kind, Context]:
        raise NotImplementedError

    def execute(self, code: Code, state: BatchedState[Code, Value, Kind, Context]) \
            -> BatchedState[Code, Value, Kind, Context]:
        raise NotImplementedError
