from typing import Generic, TypeVar, Dict
from mlprogram.interpreters.sequential_program import SequentialProgram
from mlprogram.interpreters.sequential_program import Reference


Code = TypeVar("Code")
Input = TypeVar("Input")
Value = TypeVar("Value")


class Interpreter(Generic[Code, Input, Value]):
    def eval(self, code: Code, input: Input) -> Value:
        raise NotImplementedError

    def eval_references(self, code: SequentialProgram[Code], input: Input) \
            -> Dict[Reference, Value]:
        raise NotImplementedError
