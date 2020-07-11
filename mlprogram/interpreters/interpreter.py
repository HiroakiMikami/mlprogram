from typing import Generic, TypeVar


Code = TypeVar("Code")
Value = TypeVar("Value")


class Interpreter(Generic[Code, Value]):
    def eval(self, code: Code) -> Value:
        raise NotImplementedError
