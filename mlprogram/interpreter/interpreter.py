from typing import Dict, Generic, TypeVar


Code = TypeVar("Code")
Id = TypeVar("Id")
Value = TypeVar("Value")


class Interpreter(Generic[Code, Id, Value]):
    def eval(self, env: Dict[Id, Value], code: Code) -> Dict[Id, Value]:
        raise NotImplementedError
