from typing import Generic, TypeVar, Dict
from mlprogram.interpreters.sequential_program import SequentialProgram
from mlprogram.interpreters.sequential_program import Reference


Code = TypeVar("Code")
Value = TypeVar("Value")


class Interpreter(Generic[Code, Value]):
    def eval(self, code: Code) -> Value:
        raise NotImplementedError

    def eval_references(self, code: SequentialProgram[Code]) \
            -> Dict[Reference, Value]:
        raise NotImplementedError
